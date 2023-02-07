# n = int(input("Enter an integer:"))
# print("The divisors of the number are:")
# for i in range(1, n + 1):
#     if (n % i == 0):
#         print(i)

# Original_price = 100
# Net_price = 124.7
# amount = Net_price - Original_price
#
# percentage = ((amount * 100) / Original_price)
# # print("percentage = ", end='')
#
# print(round(percentage), end='')
# print("%")


include_tax = 110
added_tax = 1.07
add_tax = 7
amount = round(include_tax / added_tax)
print(amount) #100

get_discount_amount = round(amount * add_tax / 100)
print(get_discount_amount)
total = round(amount + get_discount_amount)
print(total)
