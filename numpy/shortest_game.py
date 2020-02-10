
def solution(N, K):
    # Apply top-down recursion, which is efficient with no repetition.
    if N <= 2 or K == 0:
        # Base cases: If N is 1 or 2, or K is 0, bet N-1 times of 1 chip.  
        return N - 1

    if N % 2 == 0:
        # If N is even, bet 1 all-in, and
        # continue playing game for N//2 with K-1 all-in opportunities.
        return 1 + solution(N // 2, K - 1)
    else:
        # If N is odd, bet 1 chip, and
        # continue playing game for N-1 with K all-in opportunities.
        return 1 + solution(N - 1, K)


def main():
    # Output: 7
    N = 8
    K = 0
    print(solution(N, K))

    # Output: 6
    N = 18
    K = 2
    print(solution(N, K))

    # Output: 4
    N = 10
    K = 10
    print(solution(N, K))

    # Output: 0
    N = 1
    K = 0
    print(solution(N, K))

    # Output: 8
    N = 100
    K = 5
    print(solution(N, K))


if __name__ == '__main__':
    main()
