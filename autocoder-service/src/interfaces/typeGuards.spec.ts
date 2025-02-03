import { Test, TestingModule } from '@nestjs/testing';
import { isResponse } from './iqb.interfaces';
import { isCarrier } from './api.interfaces';

describe('TypeGuards', () => {


  it('isResponse should work', () => {
    expect(isResponse({ id: 'a', value: 'A', status: 'VALUE_CHANGED' })).toBeTruthy();
  });

  it('isResponseList should work', () => {

    expect(isResponse([
      { "id": "a", "value": "A", "status": "VALUE_CHANGED" },
      { "id": "b", "value": "B", "status": "VALUE_CHANGED" },
      { "id": "c", "value": "C", "status": "VALUE_CHANGED" },
      { "id": "d", "value": "D", "status": "VALUE_CHANGED" }
    ])).toBeTruthy();
  });

  it('isCarrier', () => {
    expect(isCarrier({ a:'1', b:'2' }, 'a', ['1'])).toBeTruthy()
    expect(isCarrier({ a:'1' }, 'a', ['1'])).toBeTruthy()
    expect(isCarrier({ a:'1' }, 'c', ['1'])).toBeFalsy()
    expect(isCarrier({ a:'1' }, 'a', ['2'])).toBeFalsy()
  });
});
