import { Test, TestingModule } from '@nestjs/testing';
import { AutocoderService } from './autocoder.service';

describe('AutocoderService', () => {
  let service: AutocoderService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [AutocoderService],
    }).compile();

    service = module.get<AutocoderService>(AutocoderService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
