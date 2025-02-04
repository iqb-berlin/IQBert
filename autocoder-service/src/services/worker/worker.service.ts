import { Injectable } from '@nestjs/common';
import { interval, Observable } from 'rxjs';
import { TasksService } from '../tasks/tasks.service';
import { DataService } from '../data/data.service';
import { IdService } from '../id.service';
import { Response } from '@iqb/responses';
import { CodingScheme } from '@iqb/responses/coding-scheme';

@Injectable()
export class WorkerService {
  private runner: Observable<number> = interval(1000);

  constructor(
    private readonly ts: TasksService,
    private readonly ds: DataService
  ) {
    this.runner
      .subscribe(async second => {
        const nextTask = this.ts.getNext();
        if (!nextTask) return;
        console.log(second, nextTask.id);

        const inputData: Response[] = nextTask.data
          .filter(task => task.type == 'input')
          .map(chunk => ds.get(chunk.id))
          .reduce((agg, chunk) => [...agg, ...chunk], []);
        try {
          const coded = (new CodingScheme(nextTask.instructions)).code(inputData);
          const id = IdService.create();
          this.ds.store(id, coded);
          nextTask.data.push({
            id,
            type: 'output'
          });
          nextTask.events.push({
            message: "Done!",
            status: "finish",
            timestamp: Date.now()
          })
        } catch (error) {
          nextTask.events.push({
            message: (error instanceof Error) ? error.message : 'unknown error',
            status: 'fail',
            timestamp: Date.now()
          });
        }
      });
  }
}
