import { ModelQueueJob } from "./models";
import {
  getAllDocuments,
  updateDocument,
} from "./helpers/firestore";

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
  try {
    const url = 'http://34.60.204.208:8000/process';
    const payload = {
        id: job.id,
        video_id: job.params.avatarVideoId,
        audio_url: job.params.audioUrl
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
    console.log('Response:', result);
    await updateStatus(job, "completed", result.output_url);
    return result;
  } catch (error) {
    console.error(`Failed to handle job ${job.id}:`, error);
    await updateStatus(job, "failed");
    return;
  }

  
};

runLoop();
