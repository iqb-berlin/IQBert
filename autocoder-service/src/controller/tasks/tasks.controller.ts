import { Body, Controller, Delete, Get, HttpException, HttpStatus, Param, Patch, Put } from '@nestjs/common';
import { TasksService } from '../../services/tasks/tasks.service';
import {
  isCarrier,
  Task,
  TaskActions,
  TaskTypes
} from '../../interfaces/api.interfaces';
import { Response as IQBVariable } from '@iqb/responses';
import { isResponse } from '../../interfaces/iqb.interfaces';
import { AutocoderService } from '../../services/autocoder/autocoder.service';


@Controller('tasks')
export class TasksController {
  constructor(
    private readonly ts: TasksService,
    private readonly as: AutocoderService
  ) {
  }

  @Get()
  getAll(): Task[] {
    return this.ts.getAll();
  }

  @Put()
  put(
    @Body() body: unknown
  ): Task {
    if (!isCarrier(body, 'type', TaskTypes)) throw new HttpException('Invalid or missing task-type.', HttpStatus.NOT_ACCEPTABLE);
    if (!('instructions' in body)) throw new HttpException('Missing coding scheme.', HttpStatus.NOT_ACCEPTABLE);
    if (this.as.validateScheme(body.instructions)) return this.ts.add(body.type, body.instructions);
    throw new HttpException('Validation of coding scheme did not work.', HttpStatus.INTERNAL_SERVER_ERROR);
  }

  @Get(':taskId')
  get(
    @Param('taskId') taskId: string
  ): Task {
    return this.ts.get(taskId);
  }

  @Patch('/:taskId')
  patch(
    @Param('taskId') taskId: string,
    @Body() body: unknown // { action: TaskAction }
  ): Task {
    if (!isCarrier(body, 'action', TaskActions)) throw new HttpException('Invalid body', HttpStatus.NOT_ACCEPTABLE);
    return this.ts.action(taskId, body.action);
  }

  @Delete('/:taskId')
  delete(
    @Param('taskId') taskId: string,
  ): void {
    this.ts.delete(taskId);
  }

  @Put('/:taskId/data')
  putData(
    @Param('taskId') taskId: string,
    @Body() body: unknown //IQBVariable[]
  ): { id: string } {
    if (!TasksController.validateDataChunk(body)) return { id: 'invalid data' };
    return this.ts.addData(taskId, body)
  }

  @Get('/:taskId/data/:chunkId')
  getData(
    @Param('taskId') taskId: string,
    @Param('chunkId') chunkId: string
  ): IQBVariable[] {
    return this.ts.getData(taskId, chunkId)
  }

  @Delete('/:taskId/data/:chunkId')
  deleteData(
    @Param('taskId') taskId: string,
    @Param('chunkId') chunkId: string
  ): void {
    this.ts.deleteData(taskId, chunkId)
  }

  static validateDataChunk(thing: unknown): thing is IQBVariable[] {
    if (!Array.isArray(thing)) {
      throw new HttpException('Not an Array.', HttpStatus.NOT_ACCEPTABLE);
    }
    const failure = thing.findIndex(r => !isResponse(r));
    if (failure > -1) {
      console.log(thing[failure]);
      throw new HttpException(`Invalid dataset nr ${failure}.`, HttpStatus.NOT_ACCEPTABLE);
    }
    return true;
  }
}
