## CUDA Benchmark

Понял по ходу дела, что если мне понадобиться алгоритм распознавания образов - лучше будет его реализовать несколько иначе и без использований нейронных сетей.
JUST Benchmark FOR CUDA.

P.S. CUDA на ноутбуке могёт ~240 млрд раз выполнить пару каких-нибудь операций

# hello world

```C++
Perceptron P1 ({
        "Знаток \"ИЛИ\"", // название используется для сохранения состояния в файл
        { 2, 3, 1 },      // количественная характеристика слоев
        // какие-то параметры
        .float_generator = random_sign_float_unit_fraction,
        .validator = validate_result
    });

// обучение таблице истинности операции OR
P1.train({
    {{true, true},{true}},
    {{true, false},{true}},
    {{false, true},{true}},
    {{false, false},{false}},
});


bool* result = P1.run({true, true});
```
