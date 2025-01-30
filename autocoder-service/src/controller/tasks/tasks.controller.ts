import { Body, Controller, Delete, Get, Param, Patch, Put } from '@nestjs/common';
import { TasksService } from '../../services/tasks/tasks.service';
import { Task, TaskAction, TaskType } from '../../interfaces/interfaces';
import { IQBVariable } from '../../interfaces/iqb.interfaces';

@Controller('tasks')
export class TasksController {
  constructor(
    private ts: TasksService
  ) {
  }

  @Get()
  get(): Task[] {
    return this.ts.getAll();
  }

  @Put()
  put(
    @Body() body: { type: TaskType }
  ): Task {
    return this.ts.add(body.type);
  }

  @Patch(':taskId')
  patch(
    @Param('taskId') taskId: string,
    @Body() body: { action: TaskAction }
  ): Task {
    return this.ts.action(taskId, body.action);
  }

  @Delete(':taskId')
  delete(
    @Param('taskId') taskId: string,
  ): void {
    this.ts.delete(taskId);
  }

  @Put(':taskId/data')
  putData(
    @Param('taskId') taskId: string,
    @Body() body: IQBVariable[]
  ): {  id: string } {
    return this.ts.addData(taskId, body)
  }

  @Get(':taskId/data/:chunkId')
  getData(
    @Param('taskId') taskId: string,
    @Param('chunkId') chunkId: string
  ): IQBVariable[] {
    return this.ts.getData(taskId, chunkId)
  }

  @Delete(':taskId/data/:chunkId')
  deleteData(
    @Param('taskId') taskId: string,
    @Param('chunkId') chunkId: string
  ): void {
    this.ts.deleteData(taskId, chunkId)
  }
}
