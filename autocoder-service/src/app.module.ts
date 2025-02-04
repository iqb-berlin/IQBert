import { DynamicModule, Module } from '@nestjs/common';
import { InfoController } from './controller/info/info.controller';
import { TasksService } from './services/tasks/tasks.service';
import { TasksController } from './controller/tasks/tasks.controller';
import { DataService } from './services/data/data.service';
import { WorkerService } from './services/worker/worker.service';
import { AutocoderService } from './services/autocoder/autocoder.service';
import * as path from 'path';

@Module({
  imports: [],
  controllers: [InfoController, TasksController],
  providers: [TasksService, DataService, WorkerService, AutocoderService],
})
export class AppModule {
  static init(): DynamicModule {
    return {
      module: AppModule,
      providers: [
        { provide: 'STORAGE_DIR', useValue: path.join(__dirname, '/../../../data') },
      ],
      exports: ['STORAGE_DIR']
    }
  }
}
