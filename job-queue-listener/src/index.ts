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
  let resultPath: string | undefined = undefined;

  try {
    console.log(job)
  } catch (error) {
    console.error(`Failed to handle job ${job.id}:`, error);
    await updateStatus(job, "failed");
    return;
  }

  await updateStatus(job, "completed", resultPath);
};

runLoop();
