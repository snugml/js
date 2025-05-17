export class KNearestNeighbors {
    constructor(n_neighbors = 3) {
        this.n_neighbors = n_neighbors;
        this.X_train = [];
        this.y_train = [];
        this.isFit = false;
    }

    fit(X, y) {
        if (!Array.isArray(X) || X.length === 0 || !Array.isArray(X[0])) {
            throw new Error('X debe ser una matriz 2D no vacía.');
        }
        if (!Array.isArray(y) || y.length !== X.length) {
            throw new Error('y debe ser un arreglo del mismo tamaño que X.');
        }

        this.X_train = X;
        this.y_train = y;
        this.isFit = true;
    }

    euclideanDistance(point1, point2) {
        return Math.sqrt(point1.reduce((sum, val, i) => sum + Math.pow(val - point2[i], 2), 0));
    }

    predict(X) {
        if (!this.isFit) {
            console.warn("El modelo no ha sido entrenado con .fit()");
            return [];
        }

        return X.map(x => {
            // Calcular todas las distancias al punto x
            const distances = this.X_train.map((x_train, i) => ({
                index: i,
                distance: this.euclideanDistance(x, x_train)
            }));

            // Ordenar por distancia
            distances.sort((a, b) => a.distance - b.distance);

            // Tomar las clases de los k vecinos más cercanos
            const kClosest = distances.slice(0, this.n_neighbors).map(d => this.y_train[d.index]);

            // Retornar la clase más frecuente (modo)
            return this._mode(kClosest);
        });
    }

    _mode(array) {
        const counts = new Map();
        for (const val of array) {
            counts.set(val, (counts.get(val) || 0) + 1);
        }
    
        let maxVal = null;
        let maxCount = -1;
        for (const [val, count] of counts.entries()) {
            if (count > maxCount) {
                maxCount = count;
                maxVal = val;
            }
        }
    
        return maxVal;
    }
}
