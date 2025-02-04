import { HttpException, HttpStatus, Inject, Injectable } from '@nestjs/common';
import * as codingSchemeSchema from '../../../../definitions/coding-scheme.schema.json';
import { Validator } from 'jsonschema';
import { AutoCodingInstructions } from '../../interfaces/api.interfaces';


@Injectable()
export class AutocoderService {
  readonly schema = codingSchemeSchema;
  private readonly validator = new Validator();

  validateScheme(codingScheme: unknown): codingScheme is AutoCodingInstructions {
    try {
      this.validator.validate(codingScheme, this.schema, { required: true, throwAll: true });
      return true;
    } catch (e) {
      throw new HttpException('Invalid Scheme: ' + e, HttpStatus.NOT_ACCEPTABLE);
    }
  }
}
