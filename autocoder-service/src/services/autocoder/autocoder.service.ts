import { HttpException, HttpStatus, Inject, Injectable } from '@nestjs/common';
import * as codingSchemeScheme from '../../../../definitions/coding-scheme.schema.json';
import { Validator } from 'jsonschema';


@Injectable()
export class AutocoderService {
  readonly schema = codingSchemeScheme;
  private readonly validator = new Validator();

  validateScheme(scheme: unknown): void {
    try {
      this.validator.validate(scheme, this.schema, { required: true, throwAll: true });
    } catch (e) {
      throw new HttpException('Invalid Scheme: ' + e, HttpStatus.NOT_ACCEPTABLE);
    }
  }
}
