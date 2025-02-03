import { DynamicModule, Module } from '@nestjs/common';
import { InfoController } from './controller/info/info.controller';
import { TasksService } from './services/tasks/tasks.service';
import { TasksController } from './controller/tasks/tasks.controller';
import { DataService } from './services/data/data.service';
import * as path from 'path';

@Module({
  imports: [],
  controllers: [InfoController, TasksController],
  providers: [TasksService, DataService],
})
export class AppModule {
  static init(): DynamicModule {
    return {
      module: AppModule,
      providers: [{ provide: 'DATA_DIR', useValue: path.join(__dirname, '/../../data') }],
      exports: ['DATA_DIR']
    }
  }
}
