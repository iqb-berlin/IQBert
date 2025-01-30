import { Module } from '@nestjs/common';
import { InfoController } from './controller/info/info.controller';
import { TasksService } from './services/tasks/tasks.service';
import { TasksController } from './controller/tasks/tasks.controller';
import { DataService } from './services/data/data.service';

@Module({
  imports: [],
  controllers: [InfoController, TasksController],
  providers: [TasksService, DataService],
})
export class AppModule {}
