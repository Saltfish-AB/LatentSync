import { InstancesClient } from "@google-cloud/compute";
import {
  GCP_INSTANCE_NAME,
  GCP_PROJECT_ID,
  GCP_ZONE,
  SA_SECRET_PATH,
} from "../configs";

export const stopVM = async (): Promise<void> => {
  const zone = GCP_ZONE;
  const instanceName = GCP_INSTANCE_NAME;

  // Check if required environment variables are set
  if (!zone || !instanceName) {
    console.log(
      `Environment variables missing: ${
        !zone ? "GCP_ZONE " : ""
      }${!instanceName ? "GCP_INSTANCE_NAME" : ""}`
    );
    return;
  }

  const computeClient = new InstancesClient({
    keyFilename: SA_SECRET_PATH,
  });

  try {
    console.log(
      `Checking status of instance: ${instanceName} in zone: ${zone}...`
    );

    // Get the current status of the instance
    const [instance] = await computeClient.get({
      project: GCP_PROJECT_ID,
      zone: zone,
      instance: instanceName,
    });

    const status: string = instance.status || "UNKNOWN";
    console.log(`Current status of instance ${instanceName}: ${status}`);

    if (status === "RUNNING") {
      console.log(`Instance is running. Proceeding to stop it...`);

      // Stop the VM if it's running
      const [operation] = await computeClient.stop({
        project: GCP_PROJECT_ID,
        zone: zone,
        instance: instanceName,
      });

      // Wait for the operation to complete
      await operation.promise();
      console.log(`Instance ${instanceName} stopped successfully.`);
    } else {
      console.log(
        `Instance ${instanceName} is not running (status: ${status}). No action taken.`
      );
    }
  } catch (err: any) {
    console.error(`Error checking or stopping the instance: ${err.message}`);
  }
};
