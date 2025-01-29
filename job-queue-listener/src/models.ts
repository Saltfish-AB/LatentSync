export interface ModelQueueJob {
  id: string;
  created_at: Date;
  last_modified: Date;
  model: string;
  status: "pending" | "running" | "completed" | "failed";
  configs: Record<string, any>;
  params: Record<string, any>;
  parentId?: string;
  clipId?: string;
  children?: Record<string, any>;
  removeLogo?: boolean;
}