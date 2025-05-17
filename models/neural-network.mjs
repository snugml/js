function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(y) {
    return y * (1 - y); // y = sigmoid(x)
}

export class MLPClassifier {
    constructor(input_nodes, hidden_nodes, output_nodes, learningRate = 0.1) {
        this.input_nodes = input_nodes;
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;
        this.lr = learningRate;

        this.weights_ih = this.randomMatrix(hidden_nodes, input_nodes);
        this.weights_ho = this.randomMatrix(output_nodes, hidden_nodes);
        this.bias_h = this.randomMatrix(hidden_nodes, 1);
        this.bias_o = this.randomMatrix(output_nodes, 1);
    }

    randomMatrix(rows, cols) {
        let arr = new Array(rows);
        for (let i = 0; i < rows; i++) {
            arr[i] = new Array(cols);
            for (let j = 0; j < cols; j++) {
                arr[i][j] = Math.random() * 2 - 1;
            }
        }
        return arr;
    }

    static dot(a, b) {
        let result = new Array(a.length);
        for (let i = 0; i < a.length; i++) {
            result[i] = new Array(b[0].length).fill(0);
            for (let j = 0; j < b[0].length; j++) {
                for (let k = 0; k < a[0].length; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    static add(a, b) {
        return a.map((row, i) => row.map((val, j) => val + b[i][j]));
    }

    static transpose(m) {
        return m[0].map((_, i) => m.map(row => row[i]));
    }

    static map(m, func) {
        return m.map(row => row.map(func));
    }

    static subtract(a, b) {
        return a.map((row, i) => row.map((val, j) => val - b[i][j]));
    }

    static multiply(a, b) {
        return a.map((row, i) => row.map((val, j) => val * b[i][j]));
    }

    static scalarMultiply(m, scalar) {
        return m.map(row => row.map(val => val * scalar));
    }

    // ---- PREDICT ----
    predict(input_array) {
        let inputs = input_array.map(v => [v]);

        let hidden = MLPClassifier.dot(this.weights_ih, inputs);
        hidden = MLPClassifier.add(hidden, this.bias_h);
        hidden = MLPClassifier.map(hidden, sigmoid);

        let outputs = MLPClassifier.dot(this.weights_ho, hidden);
        outputs = MLPClassifier.add(outputs, this.bias_o);
        outputs = MLPClassifier.map(outputs, sigmoid);

        return outputs.map(row => row[0]);
    }

    // ---- TRAIN ----
    fit(X, Y, epochs = 10000) {
        for (let e = 0; e < epochs; e++) {
            for (let i = 0; i < X.length; i++) {
                this.fitSingle(X[i], Array.isArray(Y[i]) ? Y[i] : [Y[i]]);
            }
        }
    }

    fitSingle(input_array, target_array) {
        let inputs = input_array.map(v => [v]);
        let targets = target_array.map(v => [v]);

        // FORWARD
        let hidden = MLPClassifier.dot(this.weights_ih, inputs);
        hidden = MLPClassifier.add(hidden, this.bias_h);
        hidden = MLPClassifier.map(hidden, sigmoid);

        let outputs = MLPClassifier.dot(this.weights_ho, hidden);
        outputs = MLPClassifier.add(outputs, this.bias_o);
        outputs = MLPClassifier.map(outputs, sigmoid);

        // BACKPROP
        let output_errors = MLPClassifier.subtract(targets, outputs);
        let gradients = MLPClassifier.map(outputs, sigmoidDerivative);
        gradients = MLPClassifier.multiply(gradients, output_errors);
        gradients = MLPClassifier.scalarMultiply(gradients, this.lr);

        let hidden_T = MLPClassifier.transpose(hidden);
        let weights_ho_deltas = MLPClassifier.dot(gradients, hidden_T);

        this.weights_ho = MLPClassifier.add(this.weights_ho, weights_ho_deltas);
        this.bias_o = MLPClassifier.add(this.bias_o, gradients);

        let who_T = MLPClassifier.transpose(this.weights_ho);
        let hidden_errors = MLPClassifier.dot(who_T, output_errors);

        let hidden_gradient = MLPClassifier.map(hidden, sigmoidDerivative);
        hidden_gradient = MLPClassifier.multiply(hidden_gradient, hidden_errors);
        hidden_gradient = MLPClassifier.scalarMultiply(hidden_gradient, this.lr);

        let inputs_T = MLPClassifier.transpose(inputs);
        let weights_ih_deltas = MLPClassifier.dot(hidden_gradient, inputs_T);

        this.weights_ih = MLPClassifier.add(this.weights_ih, weights_ih_deltas);
        this.bias_h = MLPClassifier.add(this.bias_h, hidden_gradient);
    }
}
