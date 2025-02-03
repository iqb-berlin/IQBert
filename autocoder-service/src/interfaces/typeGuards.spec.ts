import { Test, TestingModule } from '@nestjs/testing';
import { isResponse } from './iqb.interfaces';

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
});
