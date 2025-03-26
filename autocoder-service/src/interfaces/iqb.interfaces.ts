import { Response } from '@iqb/responses';
import { ResponseStatusType, ResponseValueType, VariableCodingData } from '@iqb/responses/coding-interfaces';

export const ResponseStatusList = ['UNSET', 'NOT_REACHED', 'DISPLAYED', 'VALUE_CHANGED', 'SOURCE_MISSING',
  'DERIVE_ERROR', 'VALUE_DERIVED', 'NO_CODING', 'INVALID', 'CODING_INCOMPLETE', 'CODING_ERROR', 'CODING_COMPLETE'];

export type AutoCodingInstructions = {
    variableCodings: VariableCodingData[];
};

export const isA = <K>(collection: string[] | readonly string[], str: unknown): str is K =>
  (typeof str === "string") &&
  (collection as readonly string[]).includes(str)

export const isArrayOf = <T>(thing: unknown, typeGuard: ((t: unknown) => t is T)): thing is T[] =>
  Array.isArray(thing) && thing.every(typeGuard);

export const isMapOf = <T>(thing: unknown, typeGuard: ((t: unknown) => t is T)): thing is {  [key: string]: T } =>
  (typeof thing === 'object') &&
  (thing != null) &&
  isArrayOf(Object.values(thing), typeGuard)

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
  (thing: unknown): thing is ResponseValueType =>
    (Array.isArray(thing) && thing.every(isResponseValueType)) ||
    (['string', 'number', 'boolean'].includes(typeof thing)) ||
    (thing == null);

export const isResponse =
  (thing: unknown): thing is Response =>
    (typeof thing === 'object') && (thing != null) &&
    ('id' in thing) && (typeof thing.id === "string") &&
    ('status' in thing) && isA<ResponseStatusType>(ResponseStatusList, thing.status) &&
    ('value' in thing) && isResponseValueType(thing.value) &&
    (!('subform' in thing) || (typeof thing.subform === 'string')) &&
    (!('code' in thing) || (typeof thing.code === 'number')) &&
    (!('score' in thing) || (typeof thing.score === 'number'));

export const isResponseList = (thing: unknown): thing is Response[] =>
  isArrayOf<Response>(thing, isResponse);

export const debugTypeguard = <T>(input: unknown, typeguard: (thing: unknown) =>  thing is T): input is T => {
  console.log('debug-typeguard:', typeguard.name);
  console.log('test-object:', input);
  const conditions = typeguard.toString().split('=>', 2)[1].split('&&');
  const firstFailingCondition = conditions
    .filter(condition => !!condition)
    .find(condition => {
      console.log('condition:', condition);
      const result = eval(`(function test(thing) { return ${condition}; })`)(input);
      console.log(result);
      return !result;
    });
  console.log('final result:', !firstFailingCondition);
  return !firstFailingCondition;
}
