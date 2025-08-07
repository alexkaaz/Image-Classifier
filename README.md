# Image-Classifier
image classifier based on EfficientNetB2 model for classifying images into N classes

Проект представляет собой классификатор изображений на основе предобученной модели EfficientNetB2. Способен классифицировать изображения по N классам и показывает
высокую точность даже при небольшом кол-ве данных.

Основу модели составляет предобученная EfficientNetB2, дополненная слоями BatchNormalization, Dropout и полносвязными слоями с регуляризацией L1/L2 для улучшения
обобщающей способности. Размер входных изображений можно изменять в зависимости от выбора версии EfficientNet (EfficientNetB2 принимает на вход изображения (260 x 260).
Обучение проводится с использованием оптимизатора Adamax и динамической настройкой learning rate через ReduceLROnPlateau. 

Проект включает удобные инструменты для работы с данными: автоматическое создание датафрейма с путями к изображениям, разделение на train/validation/test выборки и
аугментацию через ImageDataGenerator. Визуализация обучения помогает контролировать процесс.  

Проект возможно использовать не только для бинарной классификации, но и для многоклассовых задач

В тестах на датасете с 25 тысячами изображений кошек и собак модель достигла accuracy 0.9925 и loss 0.15, что
свидетельствует об ее эффективности:

<img width="579" height="43" alt="image" src="https://github.com/user-attachments/assets/d3f41be4-2e3b-43ee-adf0-e56ea61969d5" />

График accuracy:

<img width="573" height="431" alt="image" src="https://github.com/user-attachments/assets/166376db-37e8-466b-b4a4-b0bd1eca6dcb" />

График loss:

<img width="567" height="430" alt="image" src="https://github.com/user-attachments/assets/34b939f5-5695-42cf-b7f2-74a1c3295290" />
