import { HttpException, Injectable, HttpStatus, Inject } from '@nestjs/common';
import * as fs from 'fs';
import { IdService } from '../id.service';
import { Response as IQBVariable } from '@iqb/responses';



@Injectable()
export class DataService {
  constructor(
    @Inject('DATA_DIR') private dataDir: string
  ) {
  }
  add(data: IQBVariable[]): string {
    const id = IdService.create();
    fs.writeFileSync(`${this.dataDir}/${id}.json`, JSON.stringify(data), { encoding: 'utf8'});  // TODO use async
    return id;
  }

  get(chunkId: string): IQBVariable[] {
    if (!fs.existsSync(`${this.dataDir}/${chunkId}.json`)) {
      throw new HttpException(`Data chunk '${chunkId}' not found.`, HttpStatus.NOT_FOUND);
    }
    const data = fs.readFileSync(`data/${chunkId}.json`, { encoding: 'utf8'}); // TODO use async
    return JSON.parse(data) as IQBVariable[]; // TODO verify
  }

  delete(chunkId: string): void {
    if (fs.existsSync(`${this.dataDir}/${chunkId}.json`)) {
      throw new HttpException(`Data chunk not found.`, HttpStatus.NOT_FOUND);
    }
    fs.rmSync(`${this.dataDir}/${chunkId}.json`); // TODO use async
  }

  restore(): string[] {
    return fs.readdirSync(`${this.dataDir}/`)
      .filter(file => file.endsWith('.json'))
      .map(file => {
        console.log(file);
        return file.substring(0, file.indexOf('.json'));
     });
  }
}
