import unittest
from unittest.mock import MagicMock, call
from jaims.agent import JAImsAgent
from jaims.entities import (
    JAImsMessage,
    JAImsMessageRole,
)

from jaims.interfaces import JAImsLLMInterface, JAImsHistoryManager, JAImsToolManager


class TestJAImsAgent(unittest.TestCase):
    def setUp(self):
        self.llm_interface = MagicMock(spec=JAImsLLMInterface)
        self.history_manager = MagicMock(spec=JAImsHistoryManager)
        self.tool_manager = MagicMock(spec=JAImsToolManager)
        self.tools = MagicMock()
        self.mock_message = JAImsMessage(
            role=JAImsMessageRole.USER,
            contents=[
                "Hello",
            ],
        )

        self.mock_message_response = JAImsMessage(
            role=JAImsMessageRole.ASSISTANT,
            contents=[
                "World",
            ],
        )
        self.mock_message_response_tool_calls = JAImsMessage.tool_call_message(
            tool_calls=[MagicMock()]
        )
        self.mock_tools = MagicMock()

    def tearDown(self) -> None:
        self.llm_interface.reset_mock()
        self.history_manager.reset_mock()
        self.tool_manager.reset_mock()
        self.tools.reset_mock()

        return super().tearDown()

    def test_run_no_messages(self):
        self.llm_interface.call.return_value = self.mock_message_response

        sut = JAImsAgent(
            llm_interface=self.llm_interface,
        )

        response = sut.message()
        self.llm_interface.call.assert_called_once_with([], [], tool_constraints=None)
        self.assertEqual(response, "World")

    def test_run_messages_and_tools(self):
        self.llm_interface.call.return_value = self.mock_message_response

        sut = JAImsAgent(
            llm_interface=self.llm_interface,
            tools=self.mock_tools,
        )

        response = sut.message([self.mock_message], tool_constraints=["hello"])
        self.llm_interface.call.assert_called_once_with(
            [self.mock_message], self.mock_tools, tool_constraints=["hello"]
        )
        self.assertEqual(response, "World")

    def test_run_passes_messages_to_history_maager(self):
        mock_history_return = MagicMock()
        self.llm_interface.call.return_value = self.mock_message_response
        self.history_manager.get_messages.return_value = mock_history_return

        sut = JAImsAgent(
            llm_interface=self.llm_interface,
            history_manager=self.history_manager,
            tools=self.mock_tools,
        )

        response = sut.message([self.mock_message])
        self.assertEqual(response, "World")

        self.llm_interface.call.assert_called_once_with(
            mock_history_return, self.mock_tools, tool_constraints=None
        )

        self.history_manager.assert_has_calls(
            [
                call.add_messages([self.mock_message]),
                call.get_messages(),
                call.add_messages([self.mock_message_response]),
            ]
        )

    def test_run_passes_tool_calls_to_tool_manager_when_receiving_tool_calls(self):
        first_message_mock = MagicMock()
        tool_response_mock = MagicMock()
        self.llm_interface.call.side_effect = [
            self.mock_message_response_tool_calls,
            self.mock_message_response,
        ]
        self.tool_manager.handle_tool_calls.return_value = [tool_response_mock]

        sut = JAImsAgent(
            llm_interface=self.llm_interface,
            tool_manager=self.tool_manager,
            tools=self.mock_tools,
        )

        response = sut.message([first_message_mock])
        self.assertEqual(response, "World")

        self.tool_manager.handle_tool_calls.assert_called_once_with(
            sut, self.mock_message_response_tool_calls.tool_calls, self.mock_tools
        )

    def test_tools_are_overridden_when_passed_trough_run_method(self):
        self.llm_interface.call.return_value = self.mock_message_response_tool_calls
        self.tool_manager.handle_tool_calls.return_value = []

        sut = JAImsAgent(
            llm_interface=self.llm_interface,
            tool_manager=self.tool_manager,
            tools=self.mock_tools,
        )

        override_tools = MagicMock()

        response = sut.message([self.mock_message], [override_tools])

        self.llm_interface.call.assert_called_once_with(
            [self.mock_message], [override_tools], tool_constraints=None
        )

        self.tool_manager.handle_tool_calls.assert_called_once_with(
            sut, self.mock_message_response_tool_calls.tool_calls, [override_tools]
        )


if __name__ == "__main__":
    unittest.main()
