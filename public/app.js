const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const uploadBtn = document.getElementById('uploadBtn');
const result = document.getElementById('result');
const labelEl = document.getElementById('label');
const confEl = document.getElementById('confidence');
const status = document.getElementById('status');

let selectedFile = null;

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) {
    selectedFile = null;
    preview.innerHTML = 'No image selected';
    uploadBtn.disabled = true;
    return;
  }
  selectedFile = file;
  uploadBtn.disabled = false;
  const img = document.createElement('img');
  img.src = URL.createObjectURL(file);
  img.onload = () => URL.revokeObjectURL(img.src);
  preview.innerHTML = '';
  preview.appendChild(img);
  result.classList.add('hidden');
  status.textContent = '';
});

uploadBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  uploadBtn.disabled = true;
  status.textContent = 'Uploading & analyzing...';
  result.classList.add('hidden');

  const form = new FormData();
  form.append('image', selectedFile);

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: form
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      status.textContent = 'Error: ' + (err.error || res.statusText);
      uploadBtn.disabled = false;
      return;
    }
    const data = await res.json();
    // expected shape: { label: "real"|"fake", confidence: 0.95 }
    labelEl.textContent = `Label: ${data.label}`;
    confEl.textContent = `Confidence: ${data.confidence}`;
    result.classList.remove('hidden');
    status.textContent = 'Done';
  } catch (e) {
    console.error(e);
    status.textContent = 'Network error';
  } finally {
    uploadBtn.disabled = false;
  }
});
