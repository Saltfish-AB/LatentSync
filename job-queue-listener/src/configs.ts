import dotenv from "dotenv";

// Load .env file into process.env
dotenv.config();

type Environment = "development" | "staging" | "production";
const ENVIRONMENT = (process.env.ENV as Environment) || "development";

let GCS_BUCKET = "saltfish-public";

const GCP_PROJECT_ID = "saltfish-434012";
const SA_SECRET_PATH =
  process.env.SA_SECRET_PATH || "/home/henrikeriksson/secrets/saltfish-434012-8c642217c8e8.json";

const REMOTE_MODELS_PATH = "/home/henrikeriksson";

const GCP_ZONE = process.env.GCP_ZONE;
const GCP_INSTANCE_NAME = process.env.GCP_INSTANCE_NAME;
const LOGO_URL =
  "https://storage.googleapis.com/saltfish-public/test/Saltfish-Full.png";

export {
  ENVIRONMENT,
  GCS_BUCKET,
  GCP_PROJECT_ID,
  SA_SECRET_PATH,
  REMOTE_MODELS_PATH,
  GCP_ZONE,
  GCP_INSTANCE_NAME,
  LOGO_URL,
};
