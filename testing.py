from testing.CodificationTest import check_action, check_observation


print("testing actions")
for _ in range(10000):
  check_action()


print("testing observations")
for i in range(10000):
    print("Game nº", i)
    check_observation()

