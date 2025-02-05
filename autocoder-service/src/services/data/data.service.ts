import { HttpException, Injectable, HttpStatus, Inject } from '@nestjs/common';
import * as fs from 'fs';
import { IdService } from '../id.service';
import { Response as IQBVariable } from '@iqb/responses';
import { AutoCodingInstructions } from '../../interfaces/api.interfaces';



@Injectable()
export class DataService {
  constructor(
    @Inject('STORAGE_DIR') private storageDir: string
  ) {
    if (!fs.existsSync(`${this.storageDir}/data`)) {
      fs.mkdirSync(`${this.storageDir}/data`);
    }
  }

  add(data: IQBVariable[]): string {
    const id = IdService.create();
    fs.writeFileSync(`${this.storageDir}/data/${id}.json`, JSON.stringify(data), { encoding: 'utf8'});  // TODO use async
    return id;
  }

  get(chunkId: string): IQBVariable[] {
    if (!fs.existsSync(`${this.storageDir}/data/${chunkId}.json`)) {
      throw new HttpException(`Data chunk '${chunkId}' not found.`, HttpStatus.NOT_FOUND);
    }
    const data = fs.readFileSync(`${this.storageDir}/data/${chunkId}.json`, { encoding: 'utf8'}); // TODO use async
    return JSON.parse(data) as IQBVariable[]; // TODO verify
  }

  delete(chunkId: string): void {
    if (fs.existsSync(`${this.storageDir}/data/${chunkId}.json`)) {
      throw new HttpException(`Data chunk not found.`, HttpStatus.NOT_FOUND);
    }
    fs.rmSync(`${this.storageDir}/${chunkId}.json`); // TODO use async
  }

  restore(): string[] {
    return fs.readdirSync(`${this.storageDir}/data/`)
      .filter(file => file.endsWith('.json'))
      .map(file => {
        return file.substring(0, file.indexOf('.json'));
     });
  }

  store(id: string, coded: IQBVariable[]): void {
    fs.writeFileSync(`${this.storageDir}/data/${id}.json`, JSON.stringify(coded));
  }

  getExampleCodingScheme(): AutoCodingInstructions {
    return JSON.parse(fs.readFileSync(`${this.storageDir}/instructions/coding-scheme.json`, { encoding: 'utf8'})); // TODO use async
  }
}
