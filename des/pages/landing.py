import reflex as rx
from typing import List, Dict, Tuple, Any
import pandas as pd


class AppState(rx.State):
    msg: str = "133457799BBCDFF0"
    key: str = "A0B1C2D3E4F56789"
    process: str = "en"
    error: bool = False
    error_msg: str = ""

    valid_process: list[str] = ["en", "de"]

    # Initial Permutation Table
    IP: list[int] = [58, 50, 42, 34, 26, 18, 10, 2,
                     60, 52, 44, 36, 28, 20, 12, 4,
                     62, 54, 46, 38, 30, 22, 14, 6,
                     64, 56, 48, 40, 32, 24, 16, 8,
                     57, 49, 41, 33, 25, 17,  9, 1,
                     59, 51, 43, 35, 27, 19, 11, 3,
                     61, 53, 45, 37, 29, 21, 13, 5,
                     63, 55, 47, 39, 31, 23, 15, 7]

    # Final Permutation Table
    FP: list[int] = [40, 8, 48, 16, 56, 24, 64, 32,
                     39, 7, 47, 15, 55, 23, 63, 31,
                     38, 6, 46, 14, 54, 22, 62, 30,
                     37, 5, 45, 13, 53, 21, 61, 29,
                     36, 4, 44, 12, 52, 20, 60, 28,
                     35, 3, 43, 11, 51, 19, 59, 27,
                     34, 2, 42, 10, 50, 18, 58, 26,
                     33, 1, 41,  9, 49, 17, 57, 25]

    # Permuted Choice 1
    PC1: list[int] = [57, 49, 41, 33, 25, 17,  9,
                      1, 58, 50, 42, 34, 26, 18,
                      10,  2, 59, 51, 43, 35, 27,
                      19, 11,  3, 60, 52, 44, 36,
                      63, 55, 47, 39, 31, 23, 15,
                      7, 62, 54, 46, 38, 30, 22,
                      14,  6, 61, 53, 45, 37, 29,
                      21, 13,  5, 28, 20, 12,  4]

    # Permuted Choice 2
    PC2: list[int] = [14, 17, 11, 24,  1,  5,
                      3, 28, 15,  6, 21, 10,
                      23, 19, 12,  4, 26,  8,
                      16,  7, 27, 20, 13,  2,
                      41, 52, 31, 37, 47, 55,
                      30, 40, 51, 45, 33, 48,
                      44, 49, 39, 56, 34, 53,
                      46, 42, 50, 36, 29, 32]

    # Number of left shifts per round
    SHIFT: list[int] = [1, 1, 2, 2, 2, 2, 2, 2,
                        1, 2, 2, 2, 2, 2, 2, 1]

    E = [32, 1, 2, 3, 4, 5,
         4, 5, 6, 7, 8, 9,
         8, 9, 10, 11, 12, 13,
         12, 13, 14, 15, 16, 17,
         16, 17, 18, 19, 20, 21,
         20, 21, 22, 23, 24, 25,
         24, 25, 26, 27, 28, 29,
         28, 29, 30, 31, 32, 1]

    P = [16, 7, 20, 21, 29, 12, 28, 17,
         1, 15, 23, 26, 5, 18, 31, 10,
         2, 8, 24, 14, 32, 27, 3, 9,
         19, 13, 30, 6, 22, 11, 4, 25]

    S_BOXES = [
        # S1
        [[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
         [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
         [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
         [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],
        # S2
        [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
         [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
         [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
         [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],
        # S3
        [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
         [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
         [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
         [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],
        # S4
        [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
         [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
         [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
         [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],
        # S5
        [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
         [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
         [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
         [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],
        # S6
        [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
         [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
         [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
         [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],
        # S7
        [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
         [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
         [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
         [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],
        # S8
        [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
         [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
         [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
         [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]
    ]

    # Helper functions
    def hex_to_bin(self, hex_str: str, bits: int) -> str:
        return bin(int(hex_str, 16))[2:].zfill(bits)

    def bin_to_hex(self, bin_str: str) -> str:
        return hex(int(bin_str, 2))[2:].upper().zfill(len(bin_str) // 4)

    def permute(self, block: str, table: List[int]) -> str:
        return ''.join(block[i - 1] for i in table)

    def left_shift(self, bits: str, n: int) -> str:
        return bits[n:] + bits[:n]

    def generate_keys(self, key: str) -> Tuple[List[str], List[str], List[str]]:
        key_bin = self.hex_to_bin(key, 64)
        key_pc1 = self.permute(key_bin, self.PC1)
        C, D = key_pc1[:28], key_pc1[28:]

        subkeys, C_list, D_list = [], [], []

        for shift in self.SHIFT:
            C = self.left_shift(C, shift)
            D = self.left_shift(D, shift)
            C_list.append(C)
            D_list.append(D)
            subkey = self.permute(C + D, self.PC2)
            subkeys.append(subkey)

        return subkeys, C_list, D_list

    def xor_bits(self, a: str, b: str) -> str:
        # a and b must be same length
        length = len(a)
        return format(int(a, 2) ^ int(b, 2), f'0{length}b')

    def sbox_substitution(self, bits48: str) -> str:
        out = []
        for i in range(8):
            block6 = bits48[i * 6:(i + 1) * 6]
            row = int(block6[0] + block6[5], 2)
            col = int(block6[1:5], 2)
            val = self.S_BOXES[i][row][col]
            out.append(format(val, '04b'))
        return ''.join(out)

    def f_function(self, R: str, subkey48: str) -> str:
        # Expand R from 32 -> 48
        expanded = self.permute(R, self.E)
        # XOR with subkey
        xored = self.xor_bits(expanded, subkey48)
        # S-box substitution -> 32 bits
        sboxed = self.sbox_substitution(xored)
        # P-permutation
        return self.permute(sboxed, self.P)

    # DES
    @rx.var
    def get_des_result(self) -> dict[str, str]:
        """Return encrypted/decrypted result of the message"""
        result = self.des_encrypt.get("result", pd.DataFrame({}))
        if result.empty:
            return {}
        return {
            "cipher": result['cipher'].loc[0],
            "decrypted_cipher": result['decrypted_cipher'].loc[0],
        }

    @rx.var
    def get_round_outputs(self) -> dict[str, str]:
        """ Return keys after 2 rounds"""
        outputs = self.des_encrypt.get("round_outputs", pd.DataFrame({}))
        if outputs.empty:
            return {}
        return {
            "L2": outputs["L2"].loc[0],
            "R2": outputs["R2"].loc[0],
        }

    # Encrypt
    @rx.var
    def des_encrypt(self) -> Dict[str, pd.DataFrame]:
        try:
            if not self.msg or not self.key:
                # Empty dataframe
                return {}

            msg_bin = self.hex_to_bin(self.msg, 64)
            msg_ip = self.permute(msg_bin, self.IP)
            L, R = msg_ip[:32], msg_ip[32:]

            round_outputs: list[tuple[str]] = []

            subkeys, C_list, D_list = self.generate_keys(self.key)
            L_list = [L]
            R_list = [R]
            for i in range(16):
                f_out = self.f_function(R, subkeys[i])
                newL = R
                newR = self.xor_bits(L, f_out)
                L, R = newL, newR
                L_list.append(L)
                R_list.append(R)
                if i == 1:
                    round_outputs.append(
                        (self.bin_to_hex(L), self.bin_to_hex(R)))

            # After 16 rounds, swap L and R and apply final permutation
            pre_output = R + L
            cipher_bin = self.permute(pre_output, self.FP)
            cipher_hex: str = self.bin_to_hex(cipher_bin)

            # Decrypted result
            decrypted_cipher = self.des_decrypt(C=cipher_hex)

            # No error commit
            self.error_raise(False)

        except (Exception) as e:
            self.error_raise(True, msg="Invalid value for hex-message")
            return {}

        return {
            "result": pd.DataFrame({
                "cipher": [cipher_hex],
                "decrypted_cipher": [decrypted_cipher]
            }),
            "subkeys": pd.DataFrame({
                "i": range(len(subkeys)),
                "key_i": [self.bin_to_hex(key) for key in subkeys]
            }),
            "permutation": pd.DataFrame({
                "i": range(len(L_list)),
                "key_left": [self.bin_to_hex(key_l) for key_l in L_list],
                "key_right": [self.bin_to_hex(key_r) for key_r in R_list],
            }),
            "round_outputs": pd.DataFrame({
                "L2": [round_outputs[0][0]],
                "R2": [round_outputs[0][1]]
            })
        }

    # Decrypt
    def des_decrypt(self, C: str) -> str:
        cipher_bin = self.hex_to_bin(C, 64)
        cipher_ip = self.permute(cipher_bin, self.IP)
        L, R = cipher_ip[:32], cipher_ip[32:]

        subkeys, C_list, D_list = self.generate_keys(self.key)
        subkeys_rev = subkeys[::-1]

        for i in range(16):
            f_out = self.f_function(R, subkeys_rev[i])
            newL = R
            newR = self.xor_bits(L, f_out)
            L, R = newL, newR

        pre_output = R + L
        decrypted_cipher_bin = self.permute(pre_output, self.FP)
        decrypted_cipher = self.bin_to_hex(decrypted_cipher_bin)

        return decrypted_cipher

    @rx.event
    def error_raise(self, value: bool, msg: str=""):
        self.error = value
        self.error_msg = msg

    @rx.event
    def set_msg(self, value: str):
        if not value.isalnum() and value != "":
            self.error_raise(True, "Invalid value, K and M should contains only alpha-numeric character")
            return
        else:
            self.error_raise(False)
            self.msg = value

    @rx.event
    def set_key(self, value: str):
        if not value.isalnum() and value != "":
            self.error_raise(True, "Invalid value, K and M should contains only alpha-numeric character")
            return
        else:
            self.error_raise(False)
            self.key = value

    @rx.event
    def set_process(self, value: str):
        self.process = value


@rx.page(route="/")
def index() -> rx.Component:
    return rx.flex(
        rx.flex(
            # Input M
            rx.input(
                placeholder="Input M here...",
                default_value=AppState.msg,
                on_change=AppState.set_msg,
            ),
            # Input key
            rx.input(
                placeholder="Input Key here...",
                default_value=AppState.key,
                on_change=AppState.set_key,
            ),
            rx.text(
                rx.cond(
                    AppState.error,
                    AppState.error_msg,
                    ""
                ),
                color=rx.color('red', 9)
            ),
            # Process selection
            rx.radio(
                items=AppState.valid_process,
                direction="row",
                spacing="4",
                default_value="en",
                on_change=AppState.set_process
            ),
            rx.vstack(
                # Cipher
                rx.text(
                    rx.cond(
                        AppState.process == "en",
                        f"Result: {AppState.get_des_result.get('cipher', "N/A")}",
                        f"Result: {AppState.get_des_result.get('decrypted_cipher', "N/A")}"
                    ),
                ),
                # Result after 2 rounds
                rx.text(f"L2: {AppState.get_round_outputs.get("L2", "N/A")}"),
                rx.text(f"R2: {AppState.get_round_outputs.get("R2", "N/A")}"),
            ),

            direction="column",
            spacing="3",
            width="30%"
        ),
        # Display
        rx.flex(
            rx.data_table(
                data=AppState.des_encrypt.get("subkeys", pd.DataFrame({})),
                pagination=True,
                sort=False
            ),
            rx.data_table(
                data=AppState.des_encrypt.get("permutation", pd.DataFrame({})),
                pagination=True,
                sort=False,
            ),
            direction="row",
            spacing="2",
            width="70%"
        ),
        direction="row",
        padding="1em",
        spacing="3",
        width='100%',
    )
