import { Injectable } from '@angular/core';
import {
  isTask,
  ServiceInfo,
  Task,
  isServiceInfo,
  ResponseRow,
  isResponseRowList,
  DataChunk, isDataChunk, TaskSeed, TaskUpdate, isCoder, Coder, TaskAction
} from '../interfaces/api.interfaces';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, catchError, Observable, tap } from 'rxjs';
import { checkCondition } from '../functions/checkCondition';
import { isArrayOf } from '../interfaces/iqb.interfaces';
import { ServiceConnection } from '../interfaces/interfaces';

@Injectable({
  providedIn: 'root'
})
export class BackendService {
  connection$: BehaviorSubject<ServiceConnection | null> = new BehaviorSubject<ServiceConnection | null>(null);

  private get url(): string {
    const service = this.connection$.getValue();
    if (!service) throw new Error('No service connected');
    return service.url;
  }

  constructor(
    private readonly http: HttpClient,
  ) {
  }

  getInfo(url: string): Observable<ServiceInfo> {
    return this.http.get<ServiceInfo>(url + '/info')
      .pipe(
        checkCondition(isServiceInfo),
        tap(info => {
          this.connection$.next({ url, info });
        }),
        catchError(e => {
          this.connection$.next(null);
          throw e;
        })
      );
  }

  getTasks(): Observable<Task[]> {
    return this.http.get<Task[]>(`${this.url}/tasks`)
      .pipe(checkCondition(response => isArrayOf(response, isTask)))
  }

  getTask(taskId: string): Observable<Task> {
    return this.http.get<Task>(`${this.url}/tasks/${taskId}`)
      .pipe(checkCondition(isTask));
  }

  getTaskData(taskId: string, chunkId: string): Observable<ResponseRow[]> {
    return this.http.get<ResponseRow[]>(`${this.url}/tasks/${taskId}/data/${chunkId}`)
      .pipe(checkCondition(isResponseRowList));
  }

  putTaskData(taskId: string, fileContent: ResponseRow[]): Observable<DataChunk> {
    return this.http.put<DataChunk>(`${this.url}/tasks/${taskId}/data/`, fileContent)
      .pipe(checkCondition(isDataChunk));
  }

  postTask(taskId: string, action: TaskAction): Observable<Task> {
    return this.http.post<Task>(`${this.url}/tasks/${taskId}/${action}`, {})
      .pipe(checkCondition(isTask));
  }

  putTask(seed: TaskSeed): Observable<Task> {
    return this.http.put<Task>(`${this.url}/tasks`, seed)
      .pipe(checkCondition(isTask));
  }

  deleteTask(taskId: string): Observable<void> {
    return this.http.delete<void>(`${this.url}/tasks/${taskId}`);
  }

  patchTask(taskId: string, update: TaskUpdate): Observable<Task> {
    return this.http.patch<Task>(`${this.url}/tasks/${taskId}`, update)
      .pipe(checkCondition(isTask));
  }

  getCoders(): Observable<Coder[]> {
    return this.http.get<Task>(`${this.url}/coders/`)
      .pipe(checkCondition(r => isArrayOf<Coder>(r, isCoder)));
  }

  deleteCoder(coderId: string): Observable<void> {
    return this.http.delete<void>(`${this.url}/coders/${coderId}`);
  }
}
