import { db } from "./firebase";
import {
  CollectionReference,
  DocumentData,
  FieldValue,
  Query,
  WhereFilterOp,
  WithFieldValue,
} from "firebase-admin/firestore";

export const addDocument = async <T extends WithFieldValue<DocumentData>>(
  collection: string,
  data: T,
  id?: string
): Promise<string> => {
  const docRef = id
    ? db.collection(collection).doc(id)
    : db.collection(collection).doc();

  await docRef.set({
    ...data,
    id: docRef.id,
    created_at: FieldValue.serverTimestamp(),
    last_modified: FieldValue.serverTimestamp(),
  });

  return docRef.id;
};

export const getDocumentById = async <T extends DocumentData>(
  collection: string,
  id: string
): Promise<T | null> => {
  const docRef = db.collection(collection).doc(id);
  const doc = await docRef.get();

  if (!doc.exists) {
    return null;
  }

  return doc.data() as T;
};

interface DocumentWithSubcollections<T> {
  data: T | null;
  subcollections: { [key: string]: DocumentData[] };
}

export const getDocumentWithSubcollections = async <T extends DocumentData>(
  collection: string,
  id: string
): Promise<DocumentWithSubcollections<T> | null> => {
  try {
    const docRef = db.collection(collection).doc(id);
    const doc = await docRef.get();

    if (!doc.exists) {
      return null;
    }

    const subcollections = await docRef.listCollections();

    // Use Promise.all to run all subcollection queries in parallel
    const subcollectionsData = await Promise.all(
      subcollections.map(async (subcollection) => {
        const subcollectionDocs = await subcollection.get();
        return {
          [subcollection.id]: subcollectionDocs.docs.map((doc) => doc.data()),
        };
      })
    );

    // Combine results into a single object
    const combinedSubcollectionsData = subcollectionsData.reduce(
      (acc, current) => ({ ...acc, ...current }),
      {}
    );

    return {
      data: doc.data() as T,
      subcollections: combinedSubcollectionsData,
    };
  } catch (error) {
    console.error("Error fetching document with subcollections:", error);
    return null;
  }
};

export const updateDocument = async <T extends WithFieldValue<DocumentData>>(
  collection: string,
  id: string,
  data: Partial<T>
): Promise<void> => {
  const docRef = db.collection(collection).doc(id);

  try {
    await db.runTransaction(async (transaction) => {
      // Read the document inside the transaction
      const docSnapshot = await transaction.get(docRef);

      if (!docSnapshot.exists) {
        throw new Error(
          `Document with ID "${id}" does not exist in collection "${collection}"`
        );
      }

      // Perform the update
      transaction.update(docRef, {
        ...data,
        last_modified: FieldValue.serverTimestamp(),
      });
    });

    console.log(
      `Document with ID "${id}" updated successfully in collection "${collection}".`
    );
  } catch (error) {
    console.error(
      `Failed to update document with ID "${id}" in collection "${collection}":`,
      error
    );
    throw error; // Re-throw the error if you want it to propagate
  }
};

export const updateOrCreateDocument = async <
  T extends WithFieldValue<DocumentData>,
>(
  collection: string,
  id: string,
  data: Partial<T>
): Promise<void> => {
  const docRef = db.collection(collection).doc(id);
  await docRef.set(
    {
      ...data,
      last_modified: FieldValue.serverTimestamp(),
    },
    { merge: true }
  );
};

export const deleteDocument = async (
  collection: string,
  id: string
): Promise<void> => {
  const docRef = db.collection(collection).doc(id);
  await docRef.delete();
};

export const getAllDocuments = async <T extends DocumentData>(
  collection: string,
  whereClauses?: [
    field: string,
    operator: FirebaseFirestore.WhereFilterOp,
    value: any,
  ][],
  orderByField?: string,
  orderDirection: "asc" | "desc" = "asc"
): Promise<T[]> => {
  return db.runTransaction(async (transaction) => {
    let collectionRef: Query<DocumentData> | CollectionReference<DocumentData> =
      db.collection(collection);

    // Apply where clauses if provided
    if (whereClauses) {
      for (const [field, operator, value] of whereClauses) {
        collectionRef = collectionRef.where(field, operator, value);
      }
    }

    // Apply ordering if provided
    if (orderByField) {
      collectionRef = collectionRef.orderBy(orderByField, orderDirection);
    }

    // Get documents in the transaction
    const snapshot = await transaction.get(collectionRef);
    const documents: T[] = [];

    snapshot.forEach((doc) => {
      documents.push(doc.data() as T);
    });

    return documents;
  });
};

export const getAllDocumentsWithSubcollections = async <T extends DocumentData>(
  collection: string,
  orderByField?: string,
  orderDirection: "asc" | "desc" = "asc"
): Promise<T[]> => {
  let collectionRef: Query<DocumentData> | CollectionReference<DocumentData> =
    db.collection(collection);

  if (orderByField) {
    collectionRef = collectionRef.orderBy(orderByField, orderDirection);
  }

  const snapshot = await collectionRef.get();
  const documents: T[] = [];

  for (const doc of snapshot.docs) {
    const documentData = doc.data() as T;

    const subcollections = await doc.ref.listCollections();
    const subcollectionData: { [key: string]: any[] } = {};

    for (const subcollection of subcollections) {
      let subcollectionRef:
        | Query<DocumentData>
        | CollectionReference<DocumentData> = subcollection;

      // Apply ordering to subcollections if `orderByField` is provided
      if (orderByField) {
        subcollectionRef = subcollectionRef.orderBy(
          orderByField,
          orderDirection
        );
      }

      const subcollectionSnapshot = await subcollectionRef.get();
      const subcollectionDocs = subcollectionSnapshot.docs.map((subDoc) =>
        subDoc.data()
      );

      subcollectionData[subcollection.id] = subcollectionDocs;
    }

    // Include subcollections in the document data
    documents.push({
      ...documentData,
      subcollections: subcollectionData,
    });
  }

  return documents;
};

// Query documents with a condition
export const queryDocuments = async <T extends DocumentData>(
  collection: string,
  field: string,
  operator: WhereFilterOp,
  value: unknown
): Promise<T[]> => {
  const collectionRef = db.collection(collection);
  const querySnapshot = await collectionRef.where(field, operator, value).get();

  const results: T[] = [];
  querySnapshot.forEach((doc) => {
    results.push(doc.data() as T);
  });

  return results;
};

export const deleteDocumentWithSubcollections = async (
  collection: string,
  id: string
): Promise<void> => {
  const docRef = db.collection(collection).doc(id);
  const subcollections = await docRef.listCollections();

  for (const subcollection of subcollections) {
    const subcollectionDocs = await subcollection.get();
    for (const doc of subcollectionDocs.docs) {
      await doc.ref.delete();
    }
  }

  await docRef.delete();
};
