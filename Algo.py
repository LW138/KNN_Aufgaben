def minimize_storage_costs(b, S, L, K):
    n = len(b)  # Anzahl der Monate
    dp = [[float('inf')] * (S + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    order = [[0] * (S + 1) for _ in range(n + 1)]  # Zum Speichern der Bestellmengen

    # Fülle die DP-Tabelle
    for i in range(1, n + 1):
        for j in range(S + 1):  # Möglicher Lagerstand am Ende des Monats i
            for prev_storage in range(S + 1):  # Möglicher Lagerstand am Anfang des Monats i
                print(dp)
                print("\n\n")
                k = b[i - 1] + j - prev_storage  # Erforderliche Bestellung
                if 0 <= k <= S and prev_storage + k - b[i - 1] == j:
                    cost = dp[i-1][prev_storage] + L * j + (K if k > 0 else 0)
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        order[i][j] = k  # Speichere die Bestellmenge

    # Finde die minimalen Kosten und die zugehörigen Bestellmengen
    min_cost = min(dp[n][j] for j in range(S + 1))
    min_j = min(range(S + 1), key=lambda j: dp[n][j])

    # Rückverfolgung der Bestellmengen
    optimal_orders = [0] * n
    j = min_j
    for i in range(n, 0, -1):
        optimal_orders[i-1] = order[i][j]
        j = j + b[i-1] - optimal_orders[i-1]  # Update des Lagerstands

    return min_cost, optimal_orders


def minimize_storage_costs_rek(demand, max_storage, holding_cost, order_cost):
    num_months = len(demand)  # Anzahl der Monate
    memo = {}
    order_quantities = [[0] * (max_storage + 1) for _ in range(num_months + 1)]  # Zum Speichern der Bestellmengen

    def dp(month, end_storage):
        if month == 0:
            return 0 if end_storage == 0 else float('inf')
        if (month, end_storage) in memo:
            return memo[(month, end_storage)]

        min_cost = float('inf')
        optimal_order_quantity = 0

        for prev_storage in range(max_storage + 1):  # Möglicher Lagerstand am Anfang des Monats
            order_quantity = demand[month - 1] + end_storage - prev_storage  # Erforderliche Bestellung
            if 0 <= order_quantity <= max_storage and prev_storage + order_quantity - demand[month - 1] == end_storage:
                previous_cost = dp(month - 1, prev_storage)
                current_cost = previous_cost + holding_cost * end_storage
                if order_quantity > 0:
                    current_cost += order_cost
                if current_cost < min_cost:
                    min_cost = current_cost
                    optimal_order_quantity = order_quantity

        order_quantities[month][end_storage] = optimal_order_quantity
        memo[(month, end_storage)] = min_cost
        return min_cost

    # Finde die minimalen Kosten
    min_cost = float('inf')
    min_end_storage = 0
    for end_storage in range(max_storage + 1):
        cost = dp(num_months, end_storage)
        if cost < min_cost:
            min_cost = cost
            min_end_storage = end_storage

    # Rückverfolgung der Bestellmengen
    optimal_orders = [0] * num_months
    end_storage = min_end_storage
    for month in range(num_months, 0, -1):
        optimal_orders[month - 1] = order_quantities[month][end_storage]
        end_storage = end_storage + demand[month - 1] - optimal_orders[month - 1]  # Update des Lagerstands

    return min_cost, optimal_orders

# Beispiel-Eingabe
b = [2, 3, 2, 1]
S = 3
L = 30
K = 100

n = len(b)

# Berechne die minimalen Kosten und die Bestellmengen
result, orders = minimize_storage_costs(b, S, L, K)
print(f"Minimale Gesamtkosten: {result} €")
print("Optimale Bestellmengen pro Monat:", orders)

result = minimize_storage_costs_rek(b, S, L, K)
print(f"Minimale Gesamtkosten Rekursiv: {result} €")
