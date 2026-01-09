document.addEventListener('DOMContentLoaded', () => {
    
    const handleAnalyzerPage = () => {
        if (!document.querySelector('.analyzer-container')) {
            return;
        }

        const uploader = document.getElementById('uploader');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('file-info');
        const imagePreview = document.getElementById('image-preview');
        const previewContainer = document.getElementById('preview-container');
        const uploaderContent = document.querySelector('.uploader-content');
        
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resetBtn = document.getElementById('resetBtn');
        
        const progressSection = document.getElementById('progress-section');
        const statusText = document.getElementById('status-text');
        
        const resultCard = document.getElementById('result-card');
        const resultLabel = document.getElementById('result-label');
        const confidencePercent = document.getElementById('confidence-percent');
        const confidenceFg = document.getElementById('confidence-fg');

        let selectedFile = null;

        const resetUI = () => {
            selectedFile = null;
            fileInput.value = '';
            if (fileInfo) fileInfo.textContent = 'No file selected';
            if (uploaderContent) uploaderContent.classList.remove('hidden');
            if (previewContainer) previewContainer.classList.add('hidden');
            if (analyzeBtn) analyzeBtn.disabled = true;
            if (progressSection) progressSection.classList.add('hidden');
            if (resultCard) resultCard.classList.add('hidden');
            if (uploader) uploader.classList.remove('hidden');
            if (analyzeBtn) analyzeBtn.parentElement.classList.remove('hidden');
        };

        if (uploader) uploader.addEventListener('click', () => fileInput.click());

        if (uploader) {
            uploader.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploader.classList.add('dragover');
            });

            uploader.addEventListener('dragleave', () => {
                uploader.classList.remove('dragover');
            });

            uploader.addEventListener('drop', (e) => {
                e.preventDefault();
                uploader.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    handleFileSelect(files[0]);
                }
            });
        }

        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    handleFileSelect(file);
                }
            });
        }

        const handleFileSelect = (file) => {
            selectedFile = file;
            fileInfo.textContent = file.name;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
            };
            reader.readAsDataURL(file);

            uploaderContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
        };

        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', async () => {
                if (!selectedFile) return;

                analyzeBtn.parentElement.classList.add('hidden');
                uploader.classList.add('hidden');
                progressSection.classList.remove('hidden');
                resultCard.classList.add('hidden');

                const formData = new FormData();
                formData.append('image', selectedFile);

                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        body: formData,
                    });

                    progressSection.classList.add('hidden');

                    if (!response.ok) {
                        const err = await response.json().catch(() => ({}));
                        statusText.textContent = 'Error: ' + (err.details || err.error || response.statusText);
                        resetBtn.click();
                        return;
                    }

                    const data = await response.json();
                    displayResults(data);

                } catch (error) {
                    console.error('Prediction error:', error);
                    progressSection.classList.add('hidden');
                    statusText.textContent = 'A network error occurred.';
                    resetBtn.click();
                }
            });
        }

        if (resetBtn) resetBtn.addEventListener('click', resetUI);

        const displayResults = (data) => {
            const { label, confidence } = data;
            
            resultLabel.textContent = label === 'real' ? 'Human-Made' : 'AI-Generated';
            resultLabel.className = label === 'real' ? 'real' : 'ai';

            const percentage = (confidence * 100).toFixed(0);
            confidencePercent.textContent = `${percentage}%`;

            const circumference = 2 * Math.PI * 45; // 2 * pi * radius
            const offset = circumference - (percentage / 100) * circumference;
            confidenceFg.style.strokeDashoffset = offset;
            confidenceFg.style.stroke = label === 'real' ? 'var(--accent-green)' : 'var(--accent-red)';

            resultCard.classList.remove('hidden');
        };
    };

    const handleAuthPage = () => {
        const passwordInput = document.getElementById('password');
        const togglePasswordButton = document.getElementById('toggle-password');

        if (passwordInput && togglePasswordButton) {
            togglePasswordButton.addEventListener('click', () => {
                const isPassword = passwordInput.type === 'password';
                passwordInput.type = isPassword ? 'text' : 'password';
                togglePasswordButton.classList.toggle('visible', isPassword);
            });
        }
    };

    if (document.querySelector('.analyzer-container')) {
        handleAnalyzerPage();
    } else if (document.querySelector('.auth-container')) {
        handleAuthPage();
    }
});