import { Response } from '@iqb/responses';
import { isArrayOf, isA } from './api.interfaces';
import { ResponseStatusType, ResponseValueType } from '@iqb/responses/coding-interfaces';

export const ResponseStatusList = ['UNSET', 'NOT_REACHED', 'DISPLAYED', 'VALUE_CHANGED', 'SOURCE_MISSING',
  'DERIVE_ERROR', 'VALUE_DERIVED', 'NO_CODING', 'INVALID', 'CODING_INCOMPLETE', 'CODING_ERROR', 'CODING_COMPLETE'];

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

