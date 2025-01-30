import { Controller, Get, Req } from '@nestjs/common';
import * as packageJson from '../../../package.json';
import { ServiceInfo } from '../../interfaces/interfaces';

@Controller('info')
export class InfoController {
  @Get()
  get(@Req() request: Request): ServiceInfo {
    return {
      apiVersion: packageJson?.iqb.codingServiceApiVersion || 'unknown',
      id: packageJson.name + '@ident',
      type: packageJson.name,
      version: packageJson.version
    }
  }
}
