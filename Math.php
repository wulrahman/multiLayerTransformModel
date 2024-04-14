<?php
class Math {
    public static function matmul($a, $b) {
        // Implement matrix multiplication
        $result = [];
        for ($i = 0; $i < count($a); $i++) {
            $resultRow = [];
            for ($j = 0; $j < count($b[0]); $j++) {
            $sum = 0;
                for ($k = 0; $k < count($a[0]); $k++) {
                    $sum += $a[$i][$k] * $b[$k][$j];
                }
                $resultRow[] = $sum;
            }
            $result[] = $resultRow;
        }
        return $result;
    }
    public static function add($a, $b) {
        // Implement addition
        $result = [];
        for ($i = 0; $i < count($a); $i++) {
            $resultRow = [];
            for ($j = 0; $j < count($a[$i]); $j++) {
                $resultRow[] = $a[$i][$j] + $b[$i][$j];
            }
            $result[] = $resultRow;
        }
        return $result;

    }

    public static function addMatVector($a, $b) {
        // Implement addition
        $result = [];
        for ($i = 0; $i < count($a); $i++) {
            $resultRow = [];
            for ($j = 0; $j < count($a[$i]); $j++) {
                $resultRow[] = $a[$i][$j] + $b[$j];
            }
            $result[] = $resultRow;
        }
        return $result;

    }
    public static function relu($a) {
        // Implement ReLU
        $result = [];
        foreach ($a as $row) {
            $resultRow = [];
            foreach ($row as $value) {
                $resultRow[] = max(0, $value);
            }
            $result[] = $resultRow;
        }
        return $result;
    }
    public static function sub($a, $b) {
        // Implement subtraction
        $result = [];
        for ($i = 0; $i < count($a); $i++) {
            $resultRow = [];
            for ($j = 0; $j < count($a[$i]); $j++) {
                $resultRow[] = $a[$i][$j] - $b[$i][$j];
            }
            $result[] = $resultRow;
        }
        return $result;
    }
    
    public static function randomMatrix($rows, $cols) {
        // Generate a random matrix
        $result = [];
        for ($i = 0; $i < $rows; $i++) {
            $resultRow = [];
            for ($j = 0; $j < $cols; $j++) {
                $resultRow[] = rand() / getrandmax();
            }
            $result[] = $resultRow;
        }
        return $result;
    }

    public static function randomVector($size) {
        // Generate a random vector
        $result = [];
        for ($i = 0; $i < $size; $i++) {
            $result[] = rand() / getrandmax();
        }
        return $result;
    }
    

    public static function mulVectorValue($a, $b) {
        // Implement multiplication
        $result = [];
        for ($i = 0; $i < count($a); $i++) {
            $result[] = $a[$i] * $b;
        }
        return $result;
    }
    
    public static function reluDerivative($a) {
        // Implement ReLU derivative
        $result = [];
        foreach ($a as $row) {
            $resultRow = [];
            foreach ($row as $value) {
                $resultRow[] = $value > 0 ? 1 : 0;
            }
            $result[] = $resultRow;
        }
        return $result;
    }

    public static function softmax($a) {
        // Implement softmax
        $result = [];
        foreach ($a as $row) {
            $expValues = array_map(function($x) {
                return exp($x);
            }, $row);
            $sumExpValues = array_sum($expValues);
            $resultRow = array_map(function($x) use ($sumExpValues) {
                return $x / $sumExpValues;
            }, $expValues);
            $result[] = $resultRow;
        }
        return $result;
    }

    public static function addVector($a, $b) {
        // Implement addition
        $result = [];
        for ($i = 0; $i < count($a); $i++) {
            $result[] = $a[$i] + $b;
        }
        return $result;
    }

    public static function subVector($a, $b) {
        // Implement subtraction
        $result = [];
        for ($i = 0; $i < count($a); $i++) {
            $result[] = $a[$i] - $b[$i];
        }
        return $result;
    }

    public static function transpose($a) {
        // Implement matrix transpose
        $result = [];
        for ($i = 0; $i < count($a[0]); $i++) {
            $resultRow = [];
            for ($j = 0; $j < count($a); $j++) {
                $resultRow[] = $a[$j][$i];
            }
            $result[] = $resultRow;
        }
        return $result;
    }

    public static function mulVector($a, $b) {
        // Implement element-wise multiplication
        $result = [];
        for ($i = 0; $i < count($a); $i++) {
            $result[] = $a[$i] * $b[$i];
        }
        return $result;
    }

    public static function mulMatrix($a, $b) {
        // Implement element-wise multiplication
        $result = [];
        for ($i = 0; $i < count($a); $i++) {
            $resultRow = [];
            for ($j = 0; $j < count($a[$i]); $j++) {
                $resultRow[] = $a[$i][$j] * $b[$i][$j];
            }
            $result[] = $resultRow;
        }
        return $result;
    }

    public static function sum($a) {
        // Implement sum
        $result = [];
        for ($i = 0; $i < count($a); $i++) {
            $result[] = array_sum($a[$i]);
        }
        return $result;
    }

    public static function sumVector($a) {
        // Implement sum
        return array_sum($a);
    }

    public static function subVectorValue($a, $b) {
        // Implement subtraction
        $result = [];
        for ($i = 0; $i < count($a); $i++) {
            $result[] = $a[$i] - $b;
        }
        return $result;
    }
    public static function mul($a, $b) {
        // Implement multiplication
        $result = [];
        for ($i = 0; $i < count($a); $i++) {
            $resultRow = [];
            for ($j = 0; $j < count($a[$i]); $j++) {
                $resultRow[] = $a[$i][$j] * $b;
            }
            $result[] = $resultRow;
        }
        return $result;
    }
}