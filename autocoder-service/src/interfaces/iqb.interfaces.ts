import { Response } from '@iqb/responses';
import { ResponseStatusType, ResponseValueType } from '@iqb/responses/coding-interfaces';
import { CodingScheme } from '@iqb/responses/coding-scheme';

export const ResponseStatusList = ['UNSET', 'NOT_REACHED', 'DISPLAYED', 'VALUE_CHANGED', 'SOURCE_MISSING',
  'DERIVE_ERROR', 'VALUE_DERIVED', 'NO_CODING', 'INVALID', 'CODING_INCOMPLETE', 'CODING_ERROR', 'CODING_COMPLETE'];

export type AutoCodingInstructions = {
  [Member in keyof CodingScheme]: CodingScheme[Member];
};

export function isA<K>(collection: string[] | readonly string[], str: unknown): str is K {
  return (typeof str === "string") && (collection as readonly string[]).includes(str);
}

export function isArrayOf<T>(thing: unknown, typeGuard: ((t: unknown) => t is T)): thing is T[] {
  return Array.isArray(thing) && thing.every(typeGuard);
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

export const isResponseValueType =
  (obj: unknown): obj is ResponseValueType =>
    (Array.isArray(obj) && obj.every(isResponseValueType)) ||
    (['string', 'number', 'boolean'].includes(typeof obj)) ||
    (obj == null);

export const isResponse =
  (obj: unknown): obj is Response =>
    (typeof obj === 'object') && (obj != null) &&
    ('id' in obj) && (typeof obj.id === "string") &&
    ('status' in obj) && isA<ResponseStatusType>(ResponseStatusList, obj.status) &&
    ('value' in obj) && isResponseValueType(obj.value) &&
    (!('subform' in obj) || (typeof obj.subform === 'string')) &&
    (!('code' in obj) || (typeof obj.code === 'number')) &&
    (!('score' in obj) || (typeof obj.score === 'number'));

export const isResponseList = (thing: unknown): thing is Response[] =>
  isArrayOf<Response>(thing, isResponse);
