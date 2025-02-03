import { HttpException, HttpStatus, Injectable } from '@nestjs/common';
import { Task, TaskAction, TaskEventType, TaskType } from '../../interfaces/api.interfaces';
import { CodingScheme, Response as IQBVariable } from '@iqb/responses';
import { IdService } from '../id.service';
import { DataService } from '../data/data.service';

@Injectable()
export class TasksService {
  constructor(
    private ds: DataService
  ) {
    this.restore();
  }

  private tasks: { [ id: string] : Task } = { };

  exists(id: string): boolean {
    return typeof this.tasks[id] !== 'undefined';
  }

  getAll(): Task[] {
    return Object.values(this.tasks);
  }

  static getLastEvent(task: Task): TaskEventType {
    if (!task.events.length) {
      throw new Error('Invalid task. Task must contain at least the event of it\'s creation');
    }
    return task.events.sort((a, b) => a.timestamp < b.timestamp ? 1 : -1)[0].status;
  }

  get(id: string): Task {
    if (!this.exists(id)) {
      throw new HttpException('Task not found', HttpStatus.NOT_FOUND);
    }
    return this.tasks[id];
  }

  getNext(): Task | undefined {
    return Object.values(this.tasks).find(task => TasksService.getLastEvent(task) === 'commit');
  }

  add(type: TaskType, instructions: object): Task {
    const newTask: Task = {
      id: IdService.create(),
      events: [{
        status: 'create',
        message: '',
        timestamp: Date.now()
      }],
      type,
      data: [],
      instructions,
    }
    this.tasks[newTask.id] = newTask;
    return newTask;
  }

  action(id: string, taskAction: TaskAction): Task {
    const task = this.get(id);
    const lastEvent = TasksService.getLastEvent(task);
    switch (taskAction) {
      case 'commit':
        if (lastEvent !== 'create') {
          throw new HttpException(`Can not commit ${lastEvent}ed task.`, HttpStatus.NOT_ACCEPTABLE);
        }
        task.events.push({
          status: 'commit',
          message: '',
          timestamp: Date.now()
        });
        console.log('TODO');
        break;
      case 'abort': {
        task.events.push({
          status: 'abort',
          message: '',
          timestamp: Date.now()
        });
        console.log('TODO');
        break;
      }
    }
    return this.tasks[id];
  }

  delete(id: string): void {
    const task = this.get(id);
    const lastEvent = TasksService.getLastEvent(task);
    if (!['fail', 'finish', 'abort'].includes(lastEvent)) {
      throw new HttpException(`Can not delete ${lastEvent}ed task.`, HttpStatus.FORBIDDEN);
    }
    delete this.tasks[id];
  }

  addData(taskId: string, data: IQBVariable[]): { id: string } {
    const task = this.get(taskId);
    const lastEvent = TasksService.getLastEvent(task);
    if (lastEvent !== 'create') {
      throw new HttpException(`Can not add data to ${lastEvent}ed task.`, HttpStatus.FORBIDDEN);
    }
    return { id: this.ds.add(data) };
  }

  static hasData(task: Task, chunkId: string): boolean {
    return task.data.some(d => d.id == chunkId);
  }

  getData(taskId: string, chunkId: string): IQBVariable[] {
    const task = this.get(taskId);
    if (!TasksService.hasData(task, chunkId)) {
      throw new HttpException(`Data chunk not attached to task`, HttpStatus.NOT_ACCEPTABLE);
    }
    return this.ds.get(chunkId);
  }

  deleteData(taskId: string, chunkId: string): void {
    const task = this.get(taskId);
    if (!TasksService.hasData(task, chunkId)) {
      throw new HttpException(`Data chunk not attached to task`, HttpStatus.NOT_ACCEPTABLE);
    }
    this.ds.delete(chunkId);
  }

  restore(): void {
    if (Object.keys(this.tasks).length > 0) {
      throw new Error('Only call restore on bootstrap application');
    }
    const data = this.ds.restore();
    if (!data.length) return;
    const id = '__orphaned_data__';
    this.tasks[id] = {
      data: data.map(chunkId => ({
        id: chunkId,
        type: 'input'
      })),
      events: [{
        status: 'create',
        message: 'automatically created because data dir was not empty on startup',
        timestamp: Date.now()
      }],
      id,
      type: 'undefined',
      instructions: new CodingScheme([])
    }
  }
}
