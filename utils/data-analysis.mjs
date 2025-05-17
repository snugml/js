
export class CSV {
    constructor() {
        this.data = [];
    }

    // Cargar un archivo CSV desde un File (input type="file")
    read(file, callback) {
        const reader = new FileReader();

        reader.readAsText(file);

        reader.onload = (event) => {
            const csvText = event.target.result;
            this.data = this.parse(csvText);
            if (callback) callback(this.data);
        };

        reader.onerror = (event) => {
            console.error('Error reading file:', event.target.error);
        };
    }

    // Convertir texto CSV en array de objetos
    parse(csvText) {
        const rows = csvText.trim().split('\n');
        const headers = rows[0].split(',');

        return rows.slice(1).map(row => {
            const values = row.split(',');
            const obj = {};
            headers.forEach((header, index) => {
                obj[header.trim()] = values[index].trim();
            });
            return obj;
        });
    }

    // Filtrar datos por valor numérico en una columna
    filter(column, value) {
        return this.data.filter(row => parseFloat(row[column]) > value);
    }

    // Calcular la media de una columna numérica
    mean(column) {
        const sum = this.data.reduce((acc, row) => acc + parseFloat(row[column]), 0);
        return sum / this.data.length;
    }

    // Obtener una sola columna como array
    select(column) {
        return this.data.map(row => row[column]);
    }
}
