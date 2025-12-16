#学習システム　課題コード

class original_MLP:

    @staticmethod
    def perceptron(x1, x2, w1, w2, bias):
        y = x1*w1 + x2*w2 - bias
        return 1 if y >= 0 else 0

    def forward(self, x1, x2):
        print("-" * 30)
        print(f"x1={x1}, x2={x2}")

        # ---- 1層目：条件ごとのパーセプトロン ----
        #条件を満たせば1を返す

        # x1 の条件
        p_x1_up1 = self.perceptron(x1, x2,  1, 0, 1)    #x1 >= 1
        p_x1_down2 = self.perceptron(x1, x2, -1, 0, -2) #x2 <= 2
        p_x1_up2 = self.perceptron(x1, x2,  1, 0, 2)    #x1 >= 2
        p_x1_down3 = self.perceptron(x1, x2, -1, 0, -3) #x1 <= 3
        p_x1_up3 = self.perceptron(x1, x2,  1, 0, 3)    #x1 >= 3
        p_x1_down4 = self.perceptron(x1, x2, -1, 0, -4) #x1 <= 4

        # x2 の条件
        p_x2_up1 = self.perceptron(x1, x2,  0, 1, 1)    #x2 >= 1
        p_x2_down2 = self.perceptron(x1, x2,  0,-1,-2)  #x2 <= 2
        p_x2_down3 = self.perceptron(x1, x2,  0,-1,-3)  #x2 <= 3

        # ---- 2層目：各領域の論理AND ----
        # 領域1: 1≤x1≤2 かつ 1≤x2≤3
        region1_inputs = [p_x1_up1, p_x1_down2, p_x2_up1, p_x2_down3]
        region1 = 1 if sum(region1_inputs) == 4 else 0

        # 領域2: 2≤x1≤3 かつ 1≤x2≤2
        region2_inputs = [p_x1_up2, p_x1_down3, p_x2_up1, p_x2_down2]
        region2 = 1 if sum(region2_inputs) == 4 else 0

        # 領域3: 3≤x1≤4 かつ 1≤x2≤3
        region3_inputs = [p_x1_up3, p_x1_down4, p_x2_up1, p_x2_down3]
        region3 = 1 if sum(region3_inputs) == 4 else 0

        print(f"領域1: {region1}, 領域2: {region2}, 領域3: {region3}")

        # ---- 3層目：3領域の OR ----
        output = 1 if (region1 or region2 or region3) else 0

        print(f"最終出力: {output}")
        return output


# テスト
model = original_MLP()
model.forward(1.5, 2)   # 領域1 → 1
model.forward(2.5, 1.5) # 領域2 → 1
model.forward(3.5, 2)   # 領域3 → 1
model.forward(0.5, 2)   # どこでもない → 0
model.forward(2.5, 3.5) # どこでもない → 0


"""
------------------------------
x1=1.5, x2=2
領域1: 1, 領域2: 0, 領域3: 0
最終出力: 1
------------------------------
x1=2.5, x2=1.5
領域1: 0, 領域2: 1, 領域3: 0
最終出力: 1
------------------------------
x1=3.5, x2=2
領域1: 0, 領域2: 0, 領域3: 1
最終出力: 1
------------------------------
x1=0.5, x2=2
領域1: 0, 領域2: 0, 領域3: 0
最終出力: 0
------------------------------
x1=2.5, x2=3.5
領域1: 0, 領域2: 0, 領域3: 0
最終出力: 0

"""