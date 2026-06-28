import re

with open('app/src/main/java/com/rakeshmalik/insectid/MainActivity.java', 'r', encoding='utf-8') as f:
    content = f.read()

# Fields to remove
content = re.sub(r'    private Uri photoUri;\n', '', content)
content = re.sub(r'    private android\.widget\.LinearLayout modelSelectorContainer;\n', '', content)
content = re.sub(r'    private TextView identifyModelWarning;\n', '', content)
content = re.sub(r'    private TextView selectedModelName;\n', '', content)
content = re.sub(r'    private InsectModel selectedModel;\n', '', content)
content = re.sub(r'    private View tabIdentify;\n', '', content)
content = re.sub(r'    private View tabModels;\n', '', content)
content = re.sub(r'    private View tabSettings;\n', '', content)
content = re.sub(r'    private android\.animation\.ObjectAnimator downloadIconAnimator;\n', '', content)
content = re.sub(r'    private android\.animation\.ObjectAnimator identifyIconAnimator;\n', '', content)

# 1. Update onCreate body (start from bottomNavigation)
p1_old = '''            this.bottomNavigation = findViewById(R.id.bottom_navigation);
            this.bottomNavigation.setOnItemSelectedListener(item -> {
                int itemId = item.getItemId();
                if (itemId == R.id.navigation_identify) {
                    tabIdentify.setVisibility(View.VISIBLE);
                    tabModels.setVisibility(View.GONE);
                    tabSettings.setVisibility(View.GONE);
                    if (modelDownloader != null && modelDownloader.isDownloading()) {
                        showMessage("Identify is temporarily disabled during downloads.");
                    }
                    return true;
                } else if (itemId == R.id.navigation_models) {
                    tabIdentify.setVisibility(View.GONE);
                    tabModels.setVisibility(View.VISIBLE);
                    tabSettings.setVisibility(View.GONE);
                    return true;
                } else if (itemId == R.id.navigation_settings) {
                    tabIdentify.setVisibility(View.GONE);
                    tabModels.setVisibility(View.GONE);
                    tabSettings.setVisibility(View.VISIBLE);
                    return true;
                }
                return false;
            });'''

p1_new = '''            this.bottomNavigation = findViewById(R.id.bottom_navigation);
            View tabIdentify = findViewById(R.id.tab_identify);
            View tabModels = findViewById(R.id.tab_models);
            View tabSettings = findViewById(R.id.tab_settings);
            
            this.uiStateManager = new UIStateManager(this, bottomNavigation, tabIdentify, tabModels, tabSettings);
            
            this.bottomNavigation.setOnItemSelectedListener(item -> {
                int itemId = item.getItemId();
                if (itemId == R.id.navigation_identify) {
                    tabIdentify.setVisibility(View.VISIBLE);
                    tabModels.setVisibility(View.GONE);
                    tabSettings.setVisibility(View.GONE);
                    if (modelDownloader != null && modelDownloader.isDownloading()) {
                        showMessage("Identify is temporarily disabled during downloads.");
                    }
                    return true;
                } else if (itemId == R.id.navigation_models) {
                    tabIdentify.setVisibility(View.GONE);
                    tabModels.setVisibility(View.VISIBLE);
                    tabSettings.setVisibility(View.GONE);
                    return true;
                } else if (itemId == R.id.navigation_settings) {
                    tabIdentify.setVisibility(View.GONE);
                    tabModels.setVisibility(View.GONE);
                    tabSettings.setVisibility(View.VISIBLE);
                    return true;
                }
                return false;
            });'''
content = content.replace(p1_old, p1_new)

p2_old = '''            if (this.btnCancelDownload != null) {
                this.btnCancelDownload.setOnClickListener(v -> {
                    modelDownloader.cancelDownload();
                    hideDownloadProgress();
                    unlockUI();
                    refreshManageModelsList();
                });
            }'''

p2_new = '''            if (this.btnCancelDownload != null) {
                this.btnCancelDownload.setOnClickListener(v -> {
                    modelDownloader.cancelDownload();
                    hideDownloadProgress();
                    uiStateManager.unlockUI();
                    refreshManageModelsList();
                });
            }
            
            this.photoPickerHelper = new PhotoPickerHelper(this, new PhotoPickerHelper.PhotoPickerCallback() {
                @Override
                public void onPhotoCropped(Uri uri) {
                    predictionCache.clear();
                    imageView.setImageURI(uri);
                    downloadModelAndRunPredictionAsync();
                }

                @Override
                public void onShowMessage(String message) {
                    showMessage(message);
                }

                @Override
                public boolean isUiLocked() {
                    return uiStateManager.isUiLocked();
                }
            });

            this.buttonPickImage.setOnClickListener(v -> {
                if (modelDownloader != null && modelDownloader.isDownloading()) {
                    showMessage("Please wait for downloads to complete.");
                    return;
                }
                photoPickerHelper.showImagePickerDialog();
            });

            android.widget.LinearLayout modelSelectorContainer = findViewById(R.id.modelSelectorContainer);
            TextView identifyModelWarning = findViewById(R.id.identifyModelWarning);
            TextView selectedModelName = findViewById(R.id.selectedModelName);

            this.modelSelectorHelper = new ModelSelectorHelper(this, modelSelectorContainer, identifyModelWarning, selectedModelName, metadataManager, modelDownloader, new ModelSelectorHelper.ModelSelectorCallback() {
                @Override
                public void onModelSelected(InsectModel model) {
                    if (photoPickerHelper.getPhotoUri() != null) {
                        runOnUiThread(() -> {
                            if (welcomeIcon != null) {
                                welcomeIcon.setVisibility(View.GONE);
                            }
                            outputText.setText("");
                        });
                        if (predictionCache.containsKey(model.getModelName())) {
                            showPredictionResponse(predictionCache.get(model.getModelName()));
                        } else {
                            downloadModelAndRunPredictionAsync();
                        }
                    }
                }

                @Override
                public boolean isUiLocked() {
                    return uiStateManager.isUiLocked();
                }

                @Override
                public void showMessage(String message) {
                    MainActivity.this.showMessage(message);
                }
            });

            this.modelSelectorHelper.initMinHeight();

            this.uiStateManager.setLockableViews(buttonPickImage, btnDownloadAll, btnCancelDownload, enabled -> {
                modelSelectorHelper.setChipGroupsEnabled(enabled);
            });'''
content = content.replace(p2_old, p2_new)

# Remove the old pick image listener and modelSelectorContainer logic
content = re.sub(r'            this\.buttonPickImage\.setOnClickListener\(v -> showImagePickerDialog\(\)\);\n', '', content)
content = re.sub(r'            this\.modelSelectorContainer = findViewById\(R\.id\.modelSelectorContainer\);\n            this\.modelSelectorContainer\.post\(\(\) -> \{.*?\n            \}\);\n            this\.identifyModelWarning = findViewById\(R\.id\.identifyModelWarning\);\n            this\.selectedModelName = findViewById\(R\.id\.selectedModelName\);\n', '', content, flags=re.DOTALL)
content = re.sub(r'            this\.tabIdentify = findViewById\(R\.id\.tab_identify\);\n            this\.tabModels = findViewById\(R\.id\.tab_models\);\n            this\.tabSettings = findViewById\(R\.id\.tab_settings\);\n', '', content)

# update populateModelSpinner calls
content = content.replace('populateModelSpinner();', 'modelSelectorHelper.populateModelSpinner();')

# 2. update methods
content = content.replace('''    public void switchToIdentifyTab() {
        runOnUiThread(() -> {
            bottomNavigation.setSelectedItemId(R.id.navigation_identify);
        });
    }''', '''    public void switchToIdentifyTab() {
        uiStateManager.switchToIdentifyTab();
    }''')

content = content.replace('''    public void showDownloadProgressContainer() {
        runOnUiThread(() -> {
            downloadProgressContainer.setVisibility(View.VISIBLE);
            startDownloadIconAnimation();
        });
    }''', '''    public void showDownloadProgressContainer() {
        runOnUiThread(() -> {
            downloadProgressContainer.setVisibility(View.VISIBLE);
            uiStateManager.startDownloadIconAnimation();
        });
    }''')

content = content.replace('''    public void switchToModelsTab() {
        runOnUiThread(() -> {
            bottomNavigation.setSelectedItemId(R.id.navigation_models);
        });
    }''', '''    public void switchToModelsTab() {
        uiStateManager.switchToModelsTab();
    }''')

# 3. Strip block of unused methods
content = re.sub(r'    private void startDownloadIconAnimation\(\) \{.*?\}\n', '', content, flags=re.DOTALL)
content = re.sub(r'    private void stopDownloadIconAnimation\(\) \{.*?\}\n', '', content, flags=re.DOTALL)
content = re.sub(r'    private void startIdentifyIconAnimation\(\) \{.*?\}\n', '', content, flags=re.DOTALL)
content = re.sub(r'    private void stopIdentifyIconAnimation\(\) \{.*?\}\n', '', content, flags=re.DOTALL)
content = re.sub(r'    private List<InsectModel> getVisibleModels\(\) \{.*?\}\n', '', content, flags=re.DOTALL)
content = re.sub(r'    private void createModelTypeSpinner\(\) \{.*?\}\n', '', content, flags=re.DOTALL)
content = re.sub(r'    private void populateModelSpinner\(\) \{.*?\}\n', '', content, flags=re.DOTALL)
content = re.sub(r'    private void onModelChipSelected\(.*?\}\n', '', content, flags=re.DOTALL)
content = re.sub(r'    private void deselectAllChipsExcept\(.*?\}\n', '', content, flags=re.DOTALL)
content = re.sub(r'    private void selectChipForModel\(.*?\}\n', '', content, flags=re.DOTALL)
content = re.sub(r'    private void setChipGroupsEnabled\(.*?\}\n', '', content, flags=re.DOTALL)
content = re.sub(r'    // Launcher for picking an image from the gallery.*?    public synchronized void lockUI\(\)', '    public synchronized void lockUI()', content, flags=re.DOTALL)

content = re.sub(r'    public synchronized void lockUI\(\) \{.*?\}\n', 
'''    public synchronized void lockUI() {
        uiStateManager.lockUI();
    }
''', content, flags=re.DOTALL)

content = re.sub(r'    public synchronized void unlockUI\(\) \{.*?\}\n', 
'''    public synchronized void unlockUI() {
        uiStateManager.unlockUI();
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        photoPickerHelper.handleActivityResult(requestCode, resultCode, data);
    }
''', content, flags=re.DOTALL)

# Thread createModelTypeSpinner replace
content = content.replace('''runOnUiThread(() -> {
                    createModelTypeSpinner();
                    showWelcome();''', '''runOnUiThread(() -> {
                    modelSelectorHelper.populateModelSpinner();
                    showWelcome();''')

# getVisibleModels -> modelSelectorHelper.getVisibleModels()
content = content.replace('for (InsectModel m : getVisibleModels()) {', 'for (InsectModel m : modelSelectorHelper.getVisibleModels()) {')
content = content.replace('List<InsectModel> currentModels = getVisibleModels();', 'List<InsectModel> currentModels = modelSelectorHelper.getVisibleModels();')
content = content.replace('List<InsectModel> modelsToDownload = getVisibleModels().stream()', 'List<InsectModel> modelsToDownload = modelSelectorHelper.getVisibleModels().stream()')

content = content.replace('stopDownloadIconAnimation();', 'uiStateManager.stopDownloadIconAnimation();')
content = content.replace('if (selectedModel != null) {', 'if (modelSelectorHelper.getSelectedModel() != null) {')
content = content.replace('downloadOrUpdateModel(selectedModel);', 'downloadOrUpdateModel(modelSelectorHelper.getSelectedModel());')

with open('app/src/main/java/com/rakeshmalik/insectid/MainActivity.java', 'w', encoding='utf-8') as f:
    f.write(content)
