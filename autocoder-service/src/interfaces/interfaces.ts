export interface ServiceInfo {
  readonly id: string;
  readonly type: string;
  readonly version: string;
  readonly apiVersion: string;
}

export const TaskTypes = ['train', 'code'] as const;
export const TaskActions = ['commit', 'abort'] as const;
export const TaskEventTypes = [ 'create', 'commit', 'start', 'fail', 'finish', 'abort' ] as const;
export const ChunkTypes = ['input', 'output'];

export type TaskType = (typeof TaskTypes[number]);
export type TaskAction = (typeof TaskActions[number]);
export type TaskEventType = (typeof TaskEventTypes[number]);
export type ChunkType = (typeof ChunkTypes[number]);

export const KeywordCollection = { TaskTypes, TaskActions, TaskEventTypes, ChunkTypes } as const;
export type KeywordCollectionType = TaskType | TaskAction | TaskEventType | ChunkType;

export interface TaskEvent {
  readonly status: TaskEventType;
  readonly message: string;
  readonly timestamp: number;
}

export interface DataChunk {
  readonly id: string;
  readonly type: ChunkType;
}

export interface Task {
  readonly id: string;
  readonly type: TaskType;
  readonly events: TaskEvent[];
  readonly data: DataChunk[]
}

export function isIn<K extends KeywordCollectionType>(collection: keyof typeof KeywordCollection, str: string): str is K {
  return (KeywordCollection[collection] as readonly string[]).includes(str);
}

export function isTaskEvent(event: object): event is taskEvent {
  return (KeywordCollection[collection] as readonly string[]).includes(str);
}

export function isTask(task: object): task is Task {
  return ('id' in task) && (typeof task.id === 'string') &&
    ('type' in task) && (typeof task.type === 'string') && isIn<TaskType>('TaskTypes', task.type) &&
    ('event' in task) && Array.isArray(task) &&
}
