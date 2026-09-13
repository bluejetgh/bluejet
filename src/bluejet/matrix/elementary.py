import torch


def rowswap(matrix, source_row, target_row):
    result =matrix.clone()

    temp = result[source_row].clone()
    result[source_row]= result[target_row]
    result[target_row] = temp

    return result


def rowscale (matrix, source_row, scaling_factor ):
    result = matrix.clone()
    result[source_row] = result[source_row] * scaling_factor
    return result


def rowreplacement(matrix, first_row, second_row, j, k):
    result = matrix.clone()
    result[second_row] = j * result[first_row] + k * result[second_row]
    return result


def rref(matrix):
    result =matrix.clone().float()

    rows,cols = result.shape
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        if result[pivot_row, col] == 0:
            swap_row = None

            for r in range(pivot_row + 1,rows):
                if result[r, col] != 0:
                    swap_row = r
                    break

            if swap_row is not None:
                result = rowswap(result, pivot_row, swap_row)
            else:
                continue

        pivot = result[pivot_row, col]

        if pivot != 1:
            result =rowscale(result,pivot_row, 1/pivot)

        for r in range(pivot_row + 1, rows):
            if result[r, col] != 0:
                factor =-result[r, col]
                result = rowreplacement(
                    result,
                    pivot_row,
                    r,
                    factor,
                    1
                )

        pivot_row += 1

    return result


if __name__ == "__main__":
    matrix = torch.tensor([
        [1., 3., 0., 0., 3.],
        [0., 0., 1., 0., 9.],
        [0., 0., 0., 1., -4.]
    ])

    print("Original matrix:")
    print(matrix)

    step1 = rowswap(matrix, 0, 1)
    print("\nAfter R1 <-> R2:")
    print(step1)

    step2 = rowscale(step1, 0, 1/3)
    print("\nAfter (1/3)R1:")
    print(step2)

    step3 = rowreplacement(step2, 0, 2, -3, 1)
    print("\nAfter R3 = -3R1 + R3:")
    print(step3)

    print("\nRREF result:")
    print(rref(matrix))