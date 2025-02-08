import { ModelQueueJob } from "./models";
import {
  getAllDocuments,
  getDocumentById,
  updateDocument,
} from "./helpers/firestore";
import { textToSpeech } from "./helpers/eleven-labs";
import path from 'path';
import { uploadFileToGCS } from "./helpers/gcs";
import { FieldValue } from "firebase-admin/firestore";
import { concatVideos } from "./helpers/ffmpeg";

const updateStatus = async (
  job: ModelQueueJob,
  status: string,
  outputUrl?: string
) => {
  const updateData: { status: string; outputUrl?: string } = { status };
  if (outputUrl) {
    updateData.outputUrl = outputUrl;
  }
  await updateDocument("latent-sync-jobs", job.id, updateData);
  if (job.clipId) {
    try {
      await updateDocument("clips", job.clipId, updateData);
    } catch (error) {
      console.error(`Failed to update clip ${job.clipId}:`, error);
    }
  }
  if(job.params.dynamicClipChildId) {
    await updateDocument(`dynamic-clips/${job.params.dynamicClipId}/start-segments`, job.params.dynamicClipChildId, updateData);
    if(status === "completed") {
      await updateDocument("dynamic-clips", job.params.dynamicClipId, {
        completedChildren: FieldValue.increment(1)
      });
      const dynamicClip = await getDocumentById("dynamic-clips", job.params.dynamicClipId);
      if(!dynamicClip) {
        return;
      }
      console.log(dynamicClip)
      if(dynamicClip.completedChildren === dynamicClip.totalChildren){
        await updateDocument("dynamic-clips", job.params.dynamicClipId, {
          status: "completed"
        });
      }
    }
  } else if(job.params.dynamicClipId){
    try {
      if(status === "completed") {
        await updateDocument("dynamic-clips", job.params.dynamicClipId, {
          outputUrl
        });
        return;
      }
      await updateDocument("dynamic-clips", job.params.dynamicClipId, updateData);
    } catch (error) {
      console.error(`Failed to update clip ${job.clipId}:`, error);
    }
  }
};

const runLoop = async () => {
    while (true) {
      const jobs = await getAllDocuments(
        "latent-sync-jobs",
        [["status", "==", "pending"]],
        "created_at",
        "asc"
      );

      if (jobs.length === 0) {
        console.log("No jobs found. Waiting for more jobs...");
        await new Promise((resolve) => setTimeout(resolve, 5000));
        continue;
      }
      await handleJob(jobs[0] as ModelQueueJob);
  }
};

const handleJob = async (job: ModelQueueJob) => {
  await updateStatus(job, "running");
  let generatedAudioUrl;
  if(job.params.elevenLabsVoiceId){
    const outputFilePath = path.resolve(__dirname, 'output.mp3');
    await textToSpeech(job.params.elevenLabsVoiceId, job.params.textPrompt, outputFilePath, job.params.nextText)
    await uploadFileToGCS(outputFilePath, `elevenlabs/${job.id}.mp3`, "audio/mpeg")
    generatedAudioUrl = `https://storage.saltfish.ai/elevenlabs/${job.id}.mp3`;
  }
  console.log(job)
  try {
    const url = 'http://localhost:8000/process';
    const payload = {
        id: job.id,
        video_id: job.params.avatarVideoId,
        audio_url: job.params.audioUrl || generatedAudioUrl,
        start_from_backwards: job.params.dynamicClipChildId ? true : false
    };

    console.log(payload)

    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const result = await response.json();
    
    if(!job.params.dynamicClipId && !job.params.dynamicClipChildId){
      await updateStatus(job, "completed", result.output_url);
    } else if(job.params.dynamicClipChildId) {
      const dynamicClip = await getDocumentById("dynamic-clips", job.params.dynamicClipId);
      if(!dynamicClip) {
        return;
      }
      const outputFilePath = path.resolve(__dirname, 'output.mp4');
      await concatVideos(result.output_url, dynamicClip.outputUrl, outputFilePath)
      await uploadFileToGCS(outputFilePath, `elevenlabs/${job.id}.mp4`, "video/mp4")
      await updateStatus(job, "completed", `https://storage.saltfish.ai/elevenlabs/${job.id}.mp4`);
    } else if(job.params.dynamicClipId) {
      const startSegments = await getAllDocuments(`dynamic-clips/${job.params.dynamicClipId}/start-segments`);
      for(const startSegment of startSegments){
        await updateDocument("latent-sync-jobs", startSegment.jobId, {status: "pending"})
      }
      await updateStatus(job, "completed", result.output_url);
    }
  } catch (error) {
    console.error(`Failed to handle job ${job.id}:`, error);
    await updateStatus(job, "failed");
    return;
  }

  
};

runLoop();
