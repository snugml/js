export class DecisionTreeClassifier {
    constructor(maxDepth = 5) {
        this.tree = null;
        this.maxDepth = maxDepth;
        this.gain = '';
    }

    // Método principal que entrena el modelo
    fit(X, y) {
        this.tree = this.buildTree(X, y, 0); // Agregamos una profundidad para controlarla
    }

    // Función recursiva que construye el árbol
    buildTree(X, y, depth) {
        const uniqueLabels = [...new Set(y)];

        // Caso base 1: Si todas las etiquetas son iguales, devuelve la etiqueta
        if (uniqueLabels.length === 1) {
            return { label: uniqueLabels[0] };
        }

        // Caso base 2: Si hemos alcanzado la profundidad máxima o no hay más atributos para dividir
        if (depth >= this.maxDepth || X[0].length === 0) {
            const majorityLabel = this.majorityLabel(y);
            return { label: majorityLabel };
        }

        // Encuentra el mejor atributo para dividir
        const bestFeatureIndex = this.bestSplit(X, y);
        if (bestFeatureIndex === -1) {
            // Si no hay una ganancia de información significativa, devolvemos la etiqueta mayoritaria
            const majorityLabel = this.majorityLabel(y);
            return { label: majorityLabel };
        }

        const bestFeatureValues = [...new Set(X.map(row => row[bestFeatureIndex]))];

        // Crear nodo del árbol
        const node = {
            featureIndex: bestFeatureIndex,
            featureValues: bestFeatureValues,
            children: []
        };

        // Dividir los datos por el valor del mejor atributo
        bestFeatureValues.forEach(value => {
            const indices = X.map((row, idx) => (row[bestFeatureIndex] === value ? idx : -1)).filter(idx => idx !== -1);
            const XSubset = indices.map(idx => X[idx].filter((_, i) => i !== bestFeatureIndex)); // Subconjunto sin el atributo elegido
            const ySubset = indices.map(idx => y[idx]);

            // Llamada recursiva para construir los nodos hijos
            const childNode = this.buildTree(XSubset, ySubset, depth + 1);
            node.children.push({ value, childNode });
        });

        return node;
    }

    // Calcular la entropía de un conjunto de etiquetas
    entropy(y) {
        const labelCounts = {};
        y.forEach(label => labelCounts[label] = (labelCounts[label] || 0) + 1);
        const total = y.length;
        let entropyValue = 0;

        Object.values(labelCounts).forEach(count => {
            const prob = count / total;
            entropyValue -= prob * Math.log2(prob);
        });

        return entropyValue;
    }

    // Calcular la ganancia de información para un atributo
    informationGain(X, y, featureIndex) {
        const featureValues = [...new Set(X.map(row => row[featureIndex]))];
        const totalEntropy = this.entropy(y); // Entropía original del conjunto completo
        let weightedEntropy = 0;

        // Para cada valor único del atributo
        featureValues.forEach(value => {
            const indices = X.map((row, idx) => (row[featureIndex] === value ? idx : -1)).filter(idx => idx !== -1);
            const ySubset = indices.map(idx => y[idx]); // Subconjunto de etiquetas correspondientes a este valor de atributo

            // Calculamos la entropía del subconjunto
            const entropyValue = this.entropy(ySubset);
            weightedEntropy += (ySubset.length / y.length) * entropyValue; // Entropía ponderada por la cantidad de instancias
        });

        // La ganancia de información es la diferencia entre la entropía original y la entropía ponderada
        return totalEntropy - weightedEntropy;
    }

    // Seleccionar el mejor atributo con la mayor ganancia de información
    bestSplit(X, y) {
        let bestGain = -Infinity;
        let bestFeatureIndex = -1;
        let iX = 0;

        for (let i = 0; i < X[0].length; i++) {
            const gain = this.informationGain(X, y, i);

            this.gain += 'Gain of column '+i+': '+gain+'<br>';
            if (gain > bestGain) {
                bestGain = gain;
                bestFeatureIndex = i;
            }
        }

        this.gain += '** Best feature: '+bestFeatureIndex+'<br>';

        // Si la ganancia de información es 0 o negativa, no dividimos más
        return bestFeatureIndex !== -1 && bestGain > 0 ? bestFeatureIndex : -1;
    }

    // Calcular la etiqueta mayoritaria
    majorityLabel(y) {
        const labelCounts = {};
        y.forEach(label => labelCounts[label] = (labelCounts[label] || 0) + 1);
        return Object.keys(labelCounts).reduce((a, b) => labelCounts[a] > labelCounts[b] ? a : b);
    }

    // Predecir las etiquetas para un conjunto de datos
    predict(X) {
        return X.map(row => this.predictRow(row, this.tree));
    }

    // Predicción recursiva para una sola fila
    predictRow(row, tree) {
        if (tree.label !== undefined) {
            return tree.label;
        }
        const value = row[tree.featureIndex];
        const childNode = tree.children.find(child => child.value === value);
        const filteredRow = row.filter((_, index) => index !== tree.featureIndex);
        return this.predictRow(filteredRow, childNode.childNode);
    }

    // Obtener todas las etiquetas de un subárbol
    getAllLabels(tree) {
        if (tree.label !== undefined) {
            return [tree.label];
        }

        let labels = [];
        tree.children.forEach(child => {
            labels = labels.concat(this.getAllLabels(child.childNode));
        });

        return labels;
    }


    
    // Función para imprimir el árbol de decisión en formato textual
    printTree(node, depth = 0) {
        let result = '';
        if (!node) return;

        const indent = '&nbsp;'.repeat(depth * 4); // Para representar la profundidad con indentación

        // Si es un nodo hoja
        if (node.label !== undefined) {
            result = `${indent}Label: ${node.label}<br>`;
        } else {
            // Si es un nodo de decisión
            result += `${indent}Feature ${node.featureIndex}:<br>`;
            node.featureValues.forEach(value => {
                result += `${indent}  -> Value: ${value}<br>`;
                result += this.printTree(node.children.find(child => child.value === value).childNode, depth + 1);
            });
        }
        return result;
    }
}
