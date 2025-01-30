export class IdService {
  static create(length: number = 12, alphabet = 'abcdefghijklmnopqrstuvwxyz'): string {
    return Array
      .from({ length })
      .map(_ => alphabet[Math.floor(Math.random() * alphabet.length)])
      .join('');
  }
}
