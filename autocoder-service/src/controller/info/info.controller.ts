import { Controller, Get, Req } from '@nestjs/common';
import * as packageJson from '../../../package.json';
import { ServiceInfo } from '../../interfaces/api.interfaces';
import { AutocoderService } from '../../services/autocoder/autocoder.service';

@Controller('info')
export class InfoController {
  constructor(
    private readonly as: AutocoderService
  ) {
  }
  @Get()
  get(@Req() request: Request): ServiceInfo {
    return {
      apiVersion: packageJson?.iqb.codingServiceApiVersion || 'unknown',
      id: packageJson.name + '@ident',
      type: packageJson.name,
      version: packageJson.version,
      instructionsSchema: this.as.schema
    }
  }
}
