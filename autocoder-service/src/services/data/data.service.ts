import { HttpException, Injectable, HttpStatus } from '@nestjs/common';
import * as fs from 'fs';
import { IdService } from '../id.service';
import { IQBVariable } from '../../interfaces/iqb.interfaces';


@Injectable()
export class DataService {
  add(data: IQBVariable[]): string {
    const id = IdService.create();
    fs.writeFileSync(`data/${id}.json`, JSON.stringify(data), { encoding: 'utf8'});  // TODO use async
    return id;
  }

  get(chunkId: string): IQBVariable[] {
    if (fs.existsSync(`data/${chunkId}.json`)) {
      throw new HttpException(`Data chunk not found.`, HttpStatus.NOT_FOUND);
    }
    const data = fs.readFileSync(`data/${chunkId}.json`, { encoding: 'utf8'}); // TODO use async
    return JSON.parse(data) as IQBVariable[]; // TODO verify
  }

  delete(chunkId: string): void {
    if (fs.existsSync(`data/${chunkId}.json`)) {
      throw new HttpException(`Data chunk not found.`, HttpStatus.NOT_FOUND);
    }
    fs.rmSync(`data/${chunkId}.json`); // TODO use async
  }
}
