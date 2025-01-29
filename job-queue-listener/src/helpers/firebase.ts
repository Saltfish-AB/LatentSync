import { SA_SECRET_PATH } from "../configs";
import admin, { ServiceAccount } from "firebase-admin";
import { getFirestore } from "firebase-admin/firestore";
import * as fs from "fs";

const jsonData: object = JSON.parse(fs.readFileSync(SA_SECRET_PATH, "utf-8"));

const app = admin.initializeApp({
  credential: admin.credential.cert(jsonData as ServiceAccount),
});

export const db = getFirestore(app, "saltfish");
