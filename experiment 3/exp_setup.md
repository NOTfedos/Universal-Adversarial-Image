**NUM_CLASSES** = from 2 to 64, step 4

**BATCH_SIZE** = equal to NUM_CLASSES

**IMAGE_SIZE** = 64

**EPOCHS** = 5000

**MASK** = MaskRandomPixels

**CRITERION** = nn.CrossEntropyLoss()

**LR** = 0.1

**SCHEDULER** = ReduceLROnPlateau (factor=0.3, patience=250, min_lr=1e-4, threshold=1e-4)

**MODEL** = ResNet50 (weights - IMAGENET1K_V2)