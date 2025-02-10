import { AutoCodingInstructions, contains, isA, isArrayOf, isResponse } from './iqb.interfaces';
import { Response } from '@iqb/responses';

export interface ServiceInfo {
  readonly id: string;
  readonly type: string;
  readonly version: string;
  readonly apiVersion: string;
  readonly instructionsSchema: JSONSchema;
}

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

export interface ResponseRow extends Response {
  readonly setId: string;
}

export interface JSONSchema {
  $id: string;
  $schema: string;
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

export function isTask(thing: unknown): thing is Task {
  return (typeof thing === 'object') && (thing != null) &&
    ('id' in thing) && (typeof thing.id === 'string') &&
    ('type' in thing) && isA<TaskType>(TaskTypes, thing.type) &&
    ('events' in thing) && isArrayOf<TaskEvent>(thing.events, isTaskEvent) &&
    ('data' in thing) && isArrayOf<DataChunk>(thing.data, isDataChunk);
}

export const isJsonSchema = (thing: unknown): thing is JSONSchema =>
  (typeof thing === 'object') && (thing != null) &&
  ('$id' in thing) && (typeof thing.$id === 'string') &&
  ('$schema' in thing) && (typeof thing.$schema === 'string');

export const isServiceInfo = (thing: unknown): thing is ServiceInfo =>
  (typeof thing == 'object') && (thing != null) &&
  ('id' in thing) && (typeof thing.id == 'string') &&
  ('type' in thing) && (typeof thing.type == 'string') &&
  ('version' in thing) && (typeof thing.version == 'string') &&
  ('apiVersion' in thing) && (typeof thing.apiVersion == 'string') &&
  ('instructionsSchema' in thing) && (typeof thing.instructionsSchema == 'object') && isJsonSchema(thing.instructionsSchema);

export const isResponseRow= (thing: unknown): thing is ResponseRow =>
  (typeof thing == 'object') && (thing != null) &&
  isResponse(thing) && contains(thing, 'setId', '');

export const isResponseRowList = (thing: unknown): thing is ResponseRow[] =>
  isArrayOf(thing, isResponseRow);
