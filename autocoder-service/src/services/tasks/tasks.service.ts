import { HttpException, HttpStatus, Injectable } from '@nestjs/common';
import { Task, TaskAction, TaskEventType, TaskType } from '../../interfaces/interfaces';
import { IQBVariable } from '../../interfaces/iqb.interfaces';
import { IdService } from '../id.service';
import { DataService } from '../data/data.service';

@Injectable()
export class TasksService {
  constructor(
    private ds: DataService
  ) {
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

  add(type: TaskType): Task {
    const newTask: Task = {
      id: IdService.create(),
      events: [{
        status: 'create',
        message: '',
        timestamp: Date.now()
      }],
      type,
      data: []
    }
    this.tasks[newTask.id] = newTask;
    return newTask;
  }

  action(id: string, type: TaskAction): Task {
    const task = this.get(id);
    const lastEvent = TasksService.getLastEvent(task);
    switch (type) {
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
    if (lastEvent !== 'start') {
      throw new HttpException(`Can not add data to ${lastEvent}ed task.`, HttpStatus.FORBIDDEN);
    }
    return { id: this.ds.add(data) };
  }

  static hasData(task: Task, chunkId: string): boolean {
    return task.data.some(d => d.id == chunkId);
  }

  getData(taskId: string, chunkId: string): IQBVariable[] {
    const task = this.get(taskId);
    if (TasksService.hasData(task, chunkId)) {
      throw new HttpException(`Data chunk not attached to task`, HttpStatus.NOT_ACCEPTABLE);
    }
    return this.ds.get(chunkId);
  }

  deleteData(taskId: string, chunkId: string): void {
    const task = this.get(taskId);
    if (TasksService.hasData(task, chunkId)) {
      throw new HttpException(`Data chunk not attached to task`, HttpStatus.NOT_ACCEPTABLE);
    }
    this.ds.delete(chunkId);
  }
}
