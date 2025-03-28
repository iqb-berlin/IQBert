import { Component, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { DatatableComponent } from '../datatable/datatable/datatable.component';
import { OptionsetComponent } from '../optionset/optionset.component';
import { ActivatedRoute } from '@angular/router';
import { concatMap, distinctUntilChanged, filter, interval, map, of, Subscription, switchMap } from 'rxjs';
import { DataService } from '../../services/data.service';
import { FormsModule } from '@angular/forms';
import { contains, isA } from '../../interfaces/iqb.interfaces';
import { MatTab, MatTabChangeEvent, MatTabGroup } from '@angular/material/tabs';
import { TabType, TaskTab } from '../../interfaces/interfaces';
import { ChunkType, ChunkTypes, DataChunk, Task } from '../../interfaces/api.interfaces';
import { StatusPipe } from '../../pipe/status.pipe';
import { DatePipe } from '@angular/common';
import { BackendService } from '../../services/backend.service';
import { UploadComponent } from '../upload/upload.component';

@Component({
  selector: 'app-task',
  imports: [
    DatatableComponent,
    OptionsetComponent,
    FormsModule,
    MatTabGroup,
    MatTab,
    StatusPipe,
    DatePipe,
    UploadComponent
  ],
  templateUrl: './task.component.html',
  styleUrl: './task.component.css'
})
export class TaskComponent implements OnInit, OnDestroy {
  @ViewChild(UploadComponent) uploadTab: UploadComponent | undefined;

  constructor(
    private readonly route: ActivatedRoute,
    protected readonly ds: DataService,
    protected readonly bs: BackendService,
  ) {
  }

  protected tabs: TaskTab[] = [];
  protected tabIndex: number = 0;
  private readonly subscriptions: { [key: string]: Subscription } = { };

  ngOnInit(): void {
    this.subscriptions['route'] = this.route.params
      .pipe(
        filter(params => contains(params, 'id', 'string')),
        map(params => params['id']),
        switchMap(taskId =>
          this.bs.connection$
            .pipe(
              filter(c => c != null),
              map(() => taskId)
            )
        ),
        concatMap(taskId => this.ds.getTask(taskId))
      ).subscribe(task => {
        this.subscriptions['polling'] = interval(1000)
          .pipe(
            filter(() => !this.tabIndex),
            switchMap(() => this.ds.task ? this.ds.getTask(this.ds.task.id) : of(null)),
            filter(t => !!t),
            distinctUntilChanged((t1: Task, t2: Task) => (StatusPipe.getStatus(t1) === StatusPipe.getStatus(t2)) && (t1.data.length === t2.data.length))
          )
          .subscribe(task => {
            const status = StatusPipe.getStatus(task);
            this.collectTabs(task);
            if (['finish', 'fail', 'abort'].includes(status)) {
              this.subscriptions['polling'].unsubscribe();
              delete this.subscriptions['polling'];
            }
          });
      });

  }

  collectTabs(task: Task): void {
    this.tabs = [
      { id: 'overview', label: 'Task', type: 'overview' },
      { id: 'config', label: 'Config', type: 'config' },
      ...task.data.map((chunk: DataChunk): TaskTab => {
        return {
          type: chunk.type,
          label: chunk.type + ': ' + chunk.id,
          id: chunk.id
        }
      })
    ];
    if (StatusPipe.getStatus(task) === 'create') {
      this.tabs.push({ id: 'add', label: task.data.length ? '+' : 'Add input Data', type: 'add' });
    }
  }

  onTabChange($event: MatTabChangeEvent) {
    if (isA<ChunkType>(ChunkTypes, this.tabs[$event.index].type)) {
      this.ds.getTaskData(this.tabs[$event.index].id);
    }
    if (this.tabs[$event.index].type === 'add') {
      if (this.uploadTab) this.uploadTab.openFileDialog();
    }
  }

  ngOnDestroy(): void {
    Object.keys(this.subscriptions)
      .forEach(subscriptionId => {
        this.subscriptions[subscriptionId].unsubscribe();
        delete this.subscriptions[subscriptionId];
      });
  }

  changeTab(tabType: TabType, id: string | null): void {
    if (!this.ds.task) return;
    this.collectTabs(this.ds.task);
    this.tabIndex = this.tabs
      .findIndex((tab: TaskTab) => tab.type === tabType && (id && tab.id === id));

    console.log(tabType, id, this.tabIndex, this.tabs[this.tabIndex]);
  }
}
