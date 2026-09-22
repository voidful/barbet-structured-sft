'use strict';
(() => {
  const data = window.BARBET_DATA;
  const select = document.getElementById('example-select');
  const zh = document.body.dataset.language === 'zh';
  const pretty = value => { try { return JSON.stringify(JSON.parse(value), null, 2); } catch { return value; } };
  function showExample() {
    const row = data.examples.rows[Number(select.value)];
    document.getElementById('example-input').textContent = row.kind === 'structured_test'
      ? row.input : `${zh ? '系統' : 'System'}: ${row.system}\n\n${zh ? '使用者' : 'User'}: ${row.input}`;
    document.getElementById('example-output').textContent = pretty(row.prediction);
    document.getElementById('example-reference').textContent = pretty(row.expected);
    const stop = row.eos ? (zh ? '自行結束' : 'EOS reached') : (zh ? '達到生成上限' : 'generation limit reached');
    const exact = row.kind === 'structured_test' ? row.canonical_exact : row.exact_string;
    const score = zh ? `完整答案相符：${exact ? '是' : '否'}` : `whole-answer match: ${exact ? 'yes' : 'no'}`;
    document.getElementById('example-meta').textContent = `${row.generated_tokens} tokens · ${stop} · ${score}`;
  }
  select.addEventListener('change', showExample);
  showExample();
  const button = document.getElementById('copy-command');
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(document.getElementById('quick-command').textContent);
      button.textContent = zh ? '已複製' : 'Copied';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(document.getElementById('quick-command'));
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      button.textContent = zh ? '請複製選取文字' : 'Copy selected text';
    }
  });
})();
