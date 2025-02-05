import { CodingScheme } from '@iqb/responses/coding-scheme';

export interface ServiceInfo {
  readonly id: string;
  readonly type: string;
  readonly version: string;
  readonly apiVersion: string;
  readonly instructionsSchema: JSONSchema;
}

export const isServiceInfo =
  (thing: unknown): thing is ServiceInfo =>
    (typeof thing == 'object') && (thing != null) &&
    ('id' in thing) && (typeof thing.id == 'string') &&
    ('type' in thing) && (typeof thing.type == 'string') &&
    ('version' in thing) && (typeof thing.version == 'string') &&
    ('apiVersion' in thing) && (typeof thing.apiVersion == 'string') &&
    ('instructionsSchema' in thing) && (typeof thing.instructionsSchema == 'object') && isJsonSchema(thing.instructionsSchema);

export const TaskTypes = ['train', 'code', 'undefined'] as const;
export const TaskActions = ['commit', 'abort'] as const;
export const TaskEventTypes = [ 'create', 'commit', 'start', 'fail', 'finish', 'abort' ] as const;
export const ChunkTypes = ['input', 'output'];

export type TaskType = (typeof TaskTypes[number]);
export type TaskAction = (typeof TaskActions[number]);
export type TaskEventType = (typeof TaskEventTypes[number]);
export type ChunkType = (typeof ChunkTypes[number]);

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
  readonly data: DataChunk[];
  readonly instructions: AutoCodingInstructions;
}

export interface JSONSchema {
  $id: string;
  $schema: string;
}

export type AutoCodingInstructions = {
  [Member in keyof CodingScheme]: CodingScheme[Member];
};


export function isA<K>(collection: string[] | readonly string[], str: unknown): str is K {
  return (typeof str === "string") && (collection as readonly string[]).includes(str);
}

export function isTaskEvent(event: unknown): event is TaskEvent {
  return (typeof event == 'object') && (event != null) &&
    ('timestamp' in event) && (typeof event.timestamp === 'number') &&
    ('status' in event) && (typeof event.status === 'string') && isA<TaskEventType>(TaskEventTypes, event.status);
}

export function isDataChunk(thing: unknown): thing is DataChunk {
  return (typeof thing === 'object') && (thing != null) &&
    ('id' in thing) && (typeof thing.id === 'string') &&
    ('type' in thing) && isA<ChunkType>(ChunkTypes, thing.type);
}

export function isArrayOf<T>(thing: unknown, typeGuard: ((t: unknown) => t is T)): thing is T[] {
  return Array.isArray(thing) && thing.every(typeGuard);
}

export function isTask(thing: unknown): thing is Task {
  return (typeof thing === 'object') && (thing != null) &&
    ('id' in thing) && (typeof thing.id === 'string') &&
    ('type' in thing) && isA<TaskType>(TaskTypes, thing.type) &&
    ('events' in thing) && isArrayOf<TaskEvent>(thing.events, isTaskEvent) &&
    ('data' in thing) && isArrayOf<DataChunk>(thing.data, isDataChunk);
}

export const contains = <Key extends PropertyKey, Z>(thing: unknown, fieldName: Key, example: Z):
  thing is Record<Key, Z> =>
    (typeof thing === 'object') && (thing != null) && (fieldName in thing) &&
    // @ts-ignore
    (typeof thing[fieldName] === typeof example);

export const isCarrier = <Key extends string, Z extends string>(thing: unknown, fieldName: Key, collection: Z[] | readonly Z[]):
  thing is { [fieldName in Key]: Z } =>
    (typeof thing === 'object') && (thing != null) && (fieldName in thing) &&
    // @ts-ignore
    (typeof thing[fieldName] === "string") && (collection as readonly string[]).includes(thing[fieldName as string]);

export const isJsonSchema = (thing: unknown): thing is JSONSchema =>
  (typeof thing === 'object') && (thing != null) &&
  ('$id' in thing) && (typeof thing.$id === 'string') &&
  ('$schema' in thing) && (typeof thing.$schema === 'string');
