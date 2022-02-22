# Классификация Симпсонов
Домашнее задание от DLS по классификации персонажей Симпсонов

Домашнее задание представляет из себя конкурс на [Kaggle](https://www.kaggle.com/c/journey-springfield/overview)

[Пример данных]()

Зачёт задания будет происходить по качеству `F1-score`, которого вы достигнете на тестовой выборке (public leaderboard) при сдаче задания в конкурс. Максимальное количество баллов за задание равняется 15. В отличие от обычных конкурсов на kaggle, в этом конкурсе в зачёт идёт ваш результат на публичной части датасета. В тестовом датасете будет 990 картинок, для которых вам будет необходимо предсказать класс.

### Задание 1. Построение нейросети

Запустить данную сеть будет вашим мини-заданием на первую неделю, чтобы было проще участвовать в соревновании.

Данная архитектура будет очень простой и нужна для того, чтобы установить базовое понимание и получить простенький сабмит на Kaggle

*Описание слоев*:

1. размерность входа: 3x224x224 
2.размерности после слоя:  8x111x111
3. 16x54x54
4. 32x26x26
5. 64x12x12
6. выход: 96x5x5

```
class SimpleCnn(nn.Module): 
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(96 * 5 * 5, n_classes)
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        logits = self.out(x)
        return logits
```

### Задание 2
А теперь самое интересное, мы сделали простенькую сверточную сеть и смогли отправить сабмит, но получившийся скор нас явно не устраивает. Надо с этим что-то сделать. 

Несколько срочныйх улучшейни для нашей сети, которые можно сделать: 

* Учим дольше и изменяем гиперпараметры сети
* learning rate, batch size, нормализация картинки и вот это всё
* Кто же так строит нейронные сети? А где пулинги и батч нормы? Надо добавлять
* Ну разве Адам наше все? [adamW](https://www.fast.ai/2018/07/02/adam-weight-decay/) для практика, [статейка для любителей](https://openreview.net/pdf?id=ryQu7f-RZ) (очень хороший анализ), [наши](https://github.com/MichaelKonobeev/adashift/) эксперименты для заинтересованных.
* Ну разве это deep learning? Вот ResNet и Inception, которые можно зафайнтьюнить под наши данные, вот это я понимаю (можно и обучить в колабе, а можно и [готовые](https://github.com/Cadene/pretrained-models.pytorch) скачать).
* Данных не очень много, можно их аугументировать и доучититься на новом датасете ( который уже будет состоять из, как  пример аугументации, перевернутых изображений)
